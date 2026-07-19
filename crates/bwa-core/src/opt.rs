//! Alignment options, mirroring bwa-mem2's `mem_opt_t` and its `mem_opt_init()` defaults.
//!
//! # What these options actually control
//!
//! bwa-mem aligns short DNA reads to a reference genome in four stages, and almost every option
//! below tunes exactly one of them:
//!
//! 1. **Seeding.** Find maximal exact matches (SMEMs) between the read and the reference using the
//!    FM index. `-k` (minimum length), `-c` (occurrence cap), `-r`/`-s` (re-seeding), `-y` (third
//!    seeding round).
//! 2. **Chaining.** Group co-linear seeds that could belong to one alignment. `-W` (minimum chain
//!    weight), `-D` (drop weak chains), `-G` (max gap between chained seeds), `-N` (chain extension
//!    cap).
//! 3. **Extension.** Banded Smith-Waterman out from each chain to get a full local alignment.
//!    `-A`/`-B`/`-O`/`-E`/`-L` (the scoring model), `-w` (band width), `-d` (Z-drop).
//! 4. **Output.** Which alignments become SAM records. `-T` (score floor), `-h` (XA hits), `-a`,
//!    `-M`, `-Y`, `-5`, `-q`, and, for paired-end data, `-U`/`-P`/`-S`/`-m`/`-I`.
//!
//! # Scoring units
//!
//! Scores are dimensionless integers in the Smith-Waterman sense. A match adds `a`; a mismatch
//! subtracts `b`; a gap of length k costs `o + e*k`. bwa reports the final score in the `AS:i:` SAM
//! tag and the second-best in `XS:i:`. Because everything is relative to `a`, changing `-A` alone
//! would silently reinterpret every other penalty, which is exactly why bwa's `update_a`
//! (`fastmap.cpp:564`) multiplies the *unset* penalties by the new `a`. See [`MemOpt::fill_scmat`]
//! for the ordering constraint that comes with it.
//!
//! # Glossary: the short C names kept on purpose
//!
//! These field names are NOT renamed, because their whole value is that they read the same here as
//! in `bwamem.h`. This table is the translation:
//!
//! | Name | Plain language |
//! |------|----------------|
//! | `a` | match reward: score ADDED for each base of the read that equals the reference |
//! | `b` | mismatch penalty: score SUBTRACTED where the two differ (stored positive, subtracted at use) |
//! | `o_del`, `o_ins` | gap OPEN penalty, charged once when a gap starts: `_del` = bases missing from the read, `_ins` = extra bases in the read |
//! | `e_del`, `e_ins` | gap EXTEND penalty, charged for each base of the gap; total gap cost is `o + e*k` |
//! | `w` | band width: how far the alignment may wander off the diagonal, so also the longest indel one extension can find |
//! | `zdrop` | how far the running score may fall below its own best before the extension is abandoned |
//! | `t` | minimum score for an alignment to be written out at all (the C spells it `T`) |
//! | `mat` | the 5x5 table of "score of base i against base j", derived from `a` and `b` |
//! | `l_pac` | length of the packed reference: all contigs concatenated, forward strand then reverse |
//! | `rid` | reference (contig) id: an index into the contig table, which is what SAM's RNAME prints |
//! | `MEM_F_*` | the behaviour bits in [`MemOpt::flag`]; see the [`flags`] module |
//!
//! A "band" is the diagonal stripe of the dynamic-programming matrix that is actually computed.
//! Restricting to a band is what makes Smith-Waterman affordable, at the cost of being unable to see
//! any alignment that strays further from the diagonal than the band is wide.
//!
//! Reading order: the [`flags`] module (behaviour bits), then [`MemOpt`] (every tunable, grouped as
//! in the four stages above), then [`MemOpt::fill_scmat`] (the one derived field, and the ordering
//! rule that governs it).

/// `MemOpt::flag` bits, mirroring bwa-mem2's `MEM_F_*` (`reference/bwa-mem2/src/bwamem.h:62-73`).
///
/// These are OR'd into [`MemOpt::flag`], which starts at 0 (`mem_opt_init`). Note that 0x1, 0x40 and
/// 0x80 are unused by bwa: the numbering is historical, not packed.
pub mod flags {
    /// Paired-end input (`-p`/two input files).
    ///
    /// Set by `-p` in the C (`fastmap.cpp:685`) and, for the two-file case, later in `main_mem`
    /// once a second input file is seen (`fastmap.cpp:953`). When set, the aligner infers the
    /// insert-size distribution per batch (`mem_pestat`), attempts pairing, and emits mate fields
    /// (RNEXT/PNEXT/TLEN, FLAG bits 0x1/0x2/0x40/0x80) instead of the SE defaults.
    pub const PE: i32 = 0x2;
    /// Skip pairing; mate rescue still runs unless `NO_RESCUE` is also set (`-P`).
    ///
    /// With `-P` each mate keeps its own best single-end alignment: bwa does not score the
    /// pair-as-a-whole and so never pays `pen_unpaired`, never rescues a mate into an implausible
    /// insert window. Biologically this is what you want when the "pairs" are not really pairs
    /// (e.g. mate-pair libraries with unknown orientation).
    pub const NOPAIRING: i32 = 0x4;
    /// Output all found alignments, secondary ones included (`-a`).
    ///
    /// Interaction: turning this on suppresses the `XA:Z` tag entirely, because the same shadowed
    /// hits are now emitted as their own FLAG 0x100 records. Putting them in both places would
    /// double-report them. See `cmd_mem.rs::finish_se`.
    pub const ALL: i32 = 0x8;
    /// Mark shadowed hits 0x100 rather than 0x800 (`-M`).
    ///
    /// Compatibility switch for tools (Picard, older GATK) that predate the supplementary-alignment
    /// FLAG 0x800 and only understand "secondary" 0x100. See the 0x10000 encoding in
    /// `cmd_mem.rs::write_aln_se` for how bwa keeps this from corrupting the emitted FLAG.
    pub const NO_MULTI: i32 = 0x10;
    /// Skip mate rescue (`-S`).
    ///
    /// Mate rescue is the pass that, when one mate aligns confidently and the other does not, runs
    /// a local Smith-Waterman of the unaligned mate against the reference window implied by the
    /// insert-size distribution. It is expensive (measured here at ~47% of PE wall time), so `-S`
    /// exists both as a speed lever and for callers who do not want positions inferred from the
    /// insert model.
    pub const NO_RESCUE: i32 = 0x20;
    /// Output the reference FASTA header in the XR tag (`-V`).
    pub const REF_HDR: i32 = 0x100;
    /// Soft-clip supplementary alignments too, instead of hard-clipping them (`-Y`).
    ///
    /// Default bwa behaviour hard-clips (`H`) supplementary records so the read's bases appear
    /// exactly once across all its records. `-Y` soft-clips (`S`) instead, so every record carries
    /// the full SEQ, which some downstream tools require at the cost of duplicated sequence.
    pub const SOFTCLIP: i32 = 0x200;
    /// Smart pairing: mates interleaved in one input file (`-p`, set together with `PE`).
    pub const SMARTPE: i32 = 0x400;
    /// For a split alignment, take the one with the smallest coordinate as primary (`-5`).
    ///
    /// Intended for Hi-C and similar chimeric libraries, where the 5'-most segment of the read is
    /// the biologically meaningful one regardless of which segment scored highest. bwa always sets
    /// [`KEEP_SUPP_MAPQ`] alongside it (`fastmap.cpp:690`, with that exact comment in the C).
    pub const PRIMARY5: i32 = 0x800;
    /// Do not modify the MAPQ of supplementary alignments (`-q`, implied by `-5`).
    ///
    /// Without it bwa caps every later record's MAPQ at the first record's, on the reasoning that a
    /// supplementary cannot be more confident than the primary it was split from.
    pub const KEEP_SUPP_MAPQ: i32 = 0x1000;
}

/// Alignment parameters. Field names and default values mirror bwa-mem2's `mem_opt_t`
/// (see `reference/bwa-mem2/src/bwamem.cpp::mem_opt_init`, and the struct at `bwamem.h:75`).
///
/// # Invariants
///
/// - [`mat`](Self::mat) must be consistent with `a`/`b`. Anything that mutates `a` or `b` must call
///   [`fill_scmat`](Self::fill_scmat) afterwards, and must do so *after* any `-A` rescaling, or the
///   Smith-Waterman kernel silently scores with the pre-rescale matrix.
/// - Penalties (`b`, `o_*`, `e_*`, `pen_*`) are stored as POSITIVE magnitudes and subtracted at use
///   site. Only the scoring matrix holds them negated.
/// - `mapq_coef_fac` must equal `trunc(ln(mapq_coef_len))`; see its field doc.
///
/// # Supplier
///
/// Constructed by [`Default`] (the `mem_opt_init` defaults) and then overwritten field by field
/// from the command line in `bwa-cli::cmd_mem::build_opt`. Downstream crates only ever read it.
#[derive(Debug, Clone)]
pub struct MemOpt {
    /// `-A`: score added per matching base. Default 1, positive integer.
    ///
    /// This is the unit in which every other score is denominated, so bwa treats it specially: if
    /// the user sets `-A`, `update_a` (`fastmap.cpp:564-577`) multiplies every penalty the user did
    /// *not* set by the new `a`, preserving the ratios that give bwa its tuned behaviour. Setting
    /// `-A 2` alone therefore yields `-B 8 -O 12,12 -E 2,2 -T 60 -d 200 -L 10,10 -U 34`, not a
    /// suddenly-lenient aligner. Options rescaled: `-T -d -B -O -E -L -U` (bwa's own help string
    /// says "scales options -TdBOELU", `fastmap.cpp:600`).
    pub a: i32,
    /// `-B`: penalty subtracted per mismatching base. Default 4, positive integer.
    ///
    /// The `a:b` ratio sets the divergence bwa tolerates: at the default 1:4 a mismatch costs four
    /// matches, so roughly one mismatch per 5 bp is the break-even point for extending. Rescaled by
    /// `-A` when not given explicitly.
    pub b: i32,
    /// `-O`: gap OPEN penalty, charged once per gap, for deletions (`o_del`, a gap in the read
    /// relative to the reference) and insertions (`o_ins`, extra bases in the read). Default 6,6.
    ///
    /// Total cost of a length-k gap is `o + e*k` (an affine gap model). `-O INT` sets both;
    /// `-O INT,INT` sets deletion then insertion. Rescaled by `-A` when not given.
    pub o_del: i32,
    /// Insertion half of `-O` (extra bases present in the read). Default 6. Kept as its own field
    /// because bwa scores the two gap directions independently: a deletion and an insertion of the
    /// same length need not cost the same.
    pub o_ins: i32,
    /// `-E`: gap EXTENSION penalty, charged per gap base. Default 1,1. Same `INT[,INT]` deletion/
    /// insertion split as `-O`. Rescaled by `-A` when not given.
    pub e_del: i32,
    /// Insertion half of `-E`, charged per inserted base. Default 1. With `o_ins` this makes a
    /// k-base insertion cost `o_ins + e_ins * k`.
    pub e_ins: i32,
    /// `-U`: penalty subtracted when bwa chooses to report a pair as unpaired rather than force it
    /// into the inferred insert-size window. Default 17, positive integer, paired-end only.
    ///
    /// Larger values bias towards "properly paired" calls even when the individual alignments are
    /// worse; smaller values let mates go their own way. Phred-scaled in the C's own words
    /// (`bwamem.h:79`). Rescaled by `-A` when not given. No effect under `-P` (NOPAIRING), which
    /// never scores the pair jointly.
    pub pen_unpaired: i32,
    /// `-L`: penalty applied when an alignment is clipped at the read's 5' (`pen_clip5`) or 3'
    /// (`pen_clip3`) end rather than run to the read end. Default 5,5. `INT[,INT]` as for `-O`.
    ///
    /// This biases bwa towards end-to-end alignments: clipping must "pay" for itself. Note the C's
    /// comment at `bwamem.h:80`: "This score is not deducted from the DP score", i.e. it steers the
    /// choice between local and glocal extension but does not appear in the reported `AS:i:`.
    /// Rescaled by `-A` when not given.
    pub pen_clip5: i32,
    /// 3'-end half of `-L`: the penalty for clipping at the end of the read rather than aligning
    /// through to it. Default 5. Read by the right-extension path exactly as `pen_clip5` is by the
    /// left; the two are separate so an asymmetric library (adapter read-through at one end only)
    /// can be handled.
    pub pen_clip3: i32,
    /// `-w`: band width, in bases, for banded Smith-Waterman. Default 100.
    ///
    /// Caps the indel length a single extension can discover: a gap wider than `w` falls outside
    /// the band and is never scored. Raising it finds longer indels at quadratic-ish cost. NOT
    /// rescaled by `-A` (it is a length, not a score). bwa also re-derives an effective per-region
    /// band internally (`infer_bw`), so `w` is an upper bound rather than the value always used.
    pub w: i32,
    /// `-d`: off-diagonal X-dropoff, in score units. Default 100.
    ///
    /// Extension stops early once the running score falls `zdrop` below the best score seen *and*
    /// the alignment has drifted off the diagonal. This is what keeps bwa from chaining an
    /// alignment straight through a structural breakpoint: past the breakpoint the score decays and
    /// the extension is cut, leaving a supplementary alignment instead of one long wrong CIGAR.
    /// Rescaled by `-A` when not given.
    pub zdrop: i32,
    /// `-T`: minimum alignment score to emit a record at all. Default 30, in score units.
    ///
    /// Output-only: it changes nothing about how alignments are found, only which survive to SAM
    /// (`bwamem.h:87`, "only affecting output"). A read whose every region scores below `t` is
    /// written as an unmapped FLAG 4 record. Rescaled by `-A` when not given, which is why the
    /// field is named `t` here and `T` in the C.
    pub t: i32,
    /// Behaviour flags, a bitwise-OR of [`flags`]. Default 0. See that module for each bit.
    pub flag: i32,
    /// `-k`: minimum seed length, in bases. Default 19, valid range roughly 10..read length.
    ///
    /// An exact match shorter than this is not used to seed an alignment. Lower is more sensitive
    /// (finds divergent reads) and much slower, since short seeds hit the genome far more often;
    /// higher is faster and misses reads with dense variation. 19 is bwa's tuning point for
    /// ~100 bp Illumina reads against a mammalian genome.
    pub min_seed_len: i32,
    /// `-W`: discard a chain whose seeded bases total fewer than this. Default 0 (disabled), in
    /// bases. Raised by bwa's long-read presets (`-x pacbio`/`ont2d`) to 20 or 40.
    pub min_chain_weight: i32,
    /// `-N`: cap on how many chains are extended. Default `1<<30`, i.e. effectively unlimited.
    /// Undocumented in bwa's help text but parsed (`fastmap.cpp:703`).
    pub max_chain_extend: i32,
    /// `-r`: re-seeding trigger. Default 1.5, a dimensionless multiplier (>= 1.0 in practice).
    ///
    /// After the first seeding pass, any SMEM longer than `min_seed_len * split_factor` is re-seeded
    /// from its middle to look for shorter internal seeds it may have masked. This is what lets bwa
    /// find alignments hidden inside a long repeat-spanning exact match. Larger = less re-seeding =
    /// faster and less sensitive.
    pub split_factor: f32,
    /// `-s`: re-seed only when the long SMEM occurs fewer than this many times. Default 10.
    /// Undocumented in bwa's help but parsed (`fastmap.cpp:700`). A seed that already occurs often
    /// is not worth splitting further.
    pub split_width: i32,
    /// `-c`: drop any seed occurring more than this many times in the genome. Default 500.
    ///
    /// The repeat guard. A 19-mer inside a satellite repeat can occur tens of thousands of times;
    /// extending every occurrence would dominate runtime for no gain, since none of them is
    /// distinguishable. Raising it improves sensitivity in repeats at steep cost.
    pub max_occ: i32,
    /// `-G`: do not chain two seeds more than this many bases apart on the reference. Default 10000
    /// bases. Undocumented in bwa's help but parsed (`fastmap.cpp:701`). Effectively the largest
    /// deletion a single chain (and so a single alignment record) can span.
    pub max_chain_gap: i32,
    /// `-t`: worker thread count. Default 1; bwa clamps to >= 1 (`fastmap.cpp:672`, `:808`).
    /// Affects speed only: output order and content here are thread-count independent.
    pub n_threads: i32,
    /// `-K`: input bases processed per batch. Default 10,000,000 (times `-t`, see `build_opt`).
    ///
    /// Load-balancing knob with an output-visible side effect: the insert-size distribution
    /// (`mem_pestat`) is estimated *per batch*, so batch boundaries change PE results. Fixing `-K`
    /// is therefore how you make a run reproducible across thread counts, and is a precondition of
    /// this port's byte-identity claim.
    pub chunk_size: i64,
    /// `-X`: two hits are redundant if their overlap exceeds this fraction of the shorter one.
    /// Default 0.50, range 0.0..=1.0. Undocumented in bwa's help but parsed (`fastmap.cpp:711`).
    /// Note the C does NOT record this in `opt0`, so `-X` is never rescaled and never consulted by
    /// the `-x` presets.
    pub mask_level: f32,
    /// `-D`: drop a chain whose seed coverage is below this fraction of a better overlapping
    /// chain's. Default 0.50, range 0.0..=1.0. Lower keeps more marginal chains alive into the
    /// (expensive) extension stage.
    pub drop_ratio: f32,
    /// When collecting `XA:Z` alternative hits, ignore anything scoring below this fraction of the
    /// best score. Default 0.80, range 0.0..=1.0. Not settable from the command line: bwa has no
    /// option letter for `XA_drop_ratio` (`bwamem.h:99` says "only effective for the XA tag").
    pub xa_drop_ratio: f32,
    /// Second, stricter redundancy threshold used when de-duplicating regions. Default 0.95, range
    /// 0.0..=1.0. Not settable from the command line.
    pub mask_level_redun: f32,
    /// `-Q`: length scale used by the MAPQ approximation. Default 50 (bases). Undocumented in bwa's
    /// help but parsed (`fastmap.cpp:719`). Setting it also recomputes `mapq_coef_fac`.
    pub mapq_coef_len: f64,
    /// MAPQ coefficient factor. bwa-mem2 declares this `int` and sets it to `(int)log(mapq_coef_len)`,
    /// so the fractional part is truncated (`log(50) = 3.912` becomes `3`). We keep it as an `f64`
    /// holding that already-truncated integer value; the truncation matters for borderline MAPQs.
    pub mapq_coef_fac: f64,
    /// When estimating the insert-size distribution, ignore pairs whose inferred insert exceeds
    /// this many bases. Default 10000. Not settable from the command line (`bwamem.h:103`); it
    /// guards `mem_pestat` against chimeric or mismapped pairs skewing the mean and stddev.
    pub max_ins: i32,
    /// `-m`: at most this many rounds of mate-rescue Smith-Waterman per read. Default 50.
    ///
    /// Each round tries one more candidate anchor from the mate's region list, so this bounds the
    /// worst case on reads whose mate has many plausible positions. Irrelevant when
    /// [`flags::NO_RESCUE`] is set.
    pub max_matesw: i32,
    /// `-y`: occurrence threshold for bwa's third seeding round. Default 20 (`atol`, hence i64).
    ///
    /// The third round re-queries the FM index for seeds occurring at most this many times, picking
    /// up moderately-repetitive seeds the first two rounds discarded. Raising it costs time and
    /// gains sensitivity in repeats.
    pub max_mem_intv: i64,
    /// `-h` (first value): if a read has at most this many hits scoring above
    /// `xa_drop_ratio * best`, list them all in the `XA:Z` tag instead of dropping them.
    /// Default 5. Output-only.
    ///
    /// Interaction: [`flags::ALL`] (`-a`) suppresses `XA:Z` entirely, so this becomes moot.
    pub max_xa_hits: i32,
    /// `-h` (second value): the same cap, but applied when the alignment is on an ALT contig.
    /// Default 200, deliberately far higher because an ALT haplotype is *expected* to collide with
    /// the primary assembly many times. `-h INT` sets both; `-h INT,INT` sets them separately.
    ///
    /// Note: this port's `BntSeq::Contig` carries no `is_alt` flag, so ALT contigs are never
    /// recognised and this value is currently never the one used. See also `-j` in `cmd_mem.rs`.
    pub max_xa_hits_alt: i32,
    /// 5x5 substitution matrix over the nt4 alphabet (A,C,G,T,N), row-major: `mat[i*5+j]` scores
    /// base `i` against base `j`. Diagonal = `+a`, off-diagonal = `-b`, and the entire N row and
    /// column = -1 regardless of scoring (an ambiguous base is mildly penalised, never rewarded).
    ///
    /// Derived state, not an independent option: see [`MemOpt::fill_scmat`] for the ordering rule.
    pub mat: [i8; 25],
}

impl Default for MemOpt {
    /// The `mem_opt_init()` defaults verbatim (`bwamem.cpp:107-143`), matrix included, so a
    /// `MemOpt::default()` behaves exactly like `bwa-mem2 mem` with no options.
    fn default() -> Self {
        // Bound before the struct literal because the matrix is derived from them and must be
        // built from the same two numbers that go into the fields, never from a later-edited copy.
        // These are the `mem_opt_init` values: +1 per match, -4 per mismatch.
        let a = 1i32;
        let b = 4i32;
        // Zeroed only to satisfy the initialisation check; `fill_scmat` overwrites all 25 entries
        // on the next line, so no zero ever reaches the kernel.
        let mut mat = [0i8; 25];
        fill_scmat(a, b, &mut mat);
        MemOpt {
            a,
            b,
            o_del: 6,
            o_ins: 6,
            e_del: 1,
            e_ins: 1,
            pen_unpaired: 17,
            pen_clip5: 5,
            pen_clip3: 5,
            w: 100,
            zdrop: 100,
            t: 30,
            flag: 0,
            min_seed_len: 19,
            min_chain_weight: 0,
            max_chain_extend: 1 << 30,
            split_factor: 1.5,
            split_width: 10,
            max_occ: 500,
            max_chain_gap: 10000,
            n_threads: 1,
            chunk_size: 10_000_000,
            mask_level: 0.50,
            drop_ratio: 0.50,
            xa_drop_ratio: 0.80,
            mask_level_redun: 0.95,
            mapq_coef_len: 50.0,
            // (int)log(50) = 3, matching bwa-mem2's integer `mapQ_coef_fac`.
            mapq_coef_fac: (50.0f64.ln() as i32) as f64,
            max_ins: 10000,
            max_matesw: 50,
            max_mem_intv: 20,
            max_xa_hits: 5,
            max_xa_hits_alt: 200,
            mat,
        }
    }
}

impl MemOpt {
    /// Recompute the 5x5 scoring matrix from the current `a`/`b`. bwa calls `bwa_fill_scmat` only
    /// **after** `update_a` has rescaled the penalties, so any CLI that changes `-A`/`-B` must call
    /// this last, exactly as `main_mem` does.
    ///
    /// The ordering is load-bearing, not stylistic. In the C, `main_mem` runs the getopt loop, then
    /// the `-x` preset block, then `update_a(opt, &opt0)` at `fastmap.cpp:860`, and only then
    /// `bwa_fill_scmat(opt->a, opt->b, opt->mat)` at `fastmap.cpp:863`. If the matrix were filled
    /// during option parsing instead, `-A 2` would leave `mat` holding the pre-rescale values while
    /// `o_del`/`e_del`/`t` had all doubled: the extension kernel reads `mat`, the gap logic reads
    /// the scalars, and the two would disagree. The symptom would be wrong (but plausible-looking)
    /// alignments only on runs that pass `-A`, which is exactly the kind of bug that survives a
    /// default-options test suite.
    ///
    /// # Parameters
    ///
    /// - `self`: taken by `&mut` because [`mat`](Self::mat) is written in place. Reads only `a` and
    ///   `b`, both of which must already hold their FINAL post-rescale values: that precondition is
    ///   the whole subject of the paragraph above. No other field is touched, so calling this twice
    ///   in a row is harmless (it is idempotent for a fixed `a`/`b`).
    pub fn fill_scmat(&mut self) {
        fill_scmat(self.a, self.b, &mut self.mat);
    }
}

/// Fill the 5x5 scoring matrix, mirroring bwa's `bwa_fill_scmat(a, b, mat)`.
///
/// `a` is the match score and `b` the mismatch penalty as POSITIVE magnitudes (the negation happens
/// here); `mat` is overwritten in full, so no clearing is needed. Both are narrowed to `i8`, which
/// is the C's type too (`int8_t mat[25]`): scores beyond +/-127 wrap in both implementations.
///
/// # Parameters
///
/// - `a`: match reward as a POSITIVE magnitude, i.e. [`MemOpt::a`] verbatim. Written to the four
///   diagonal entries. Must be within `i8` range (1..=127 in practice) or it wraps silently, which
///   is bwa's behaviour too and so is preserved rather than fixed.
/// - `b`: mismatch penalty as a POSITIVE magnitude, i.e. [`MemOpt::b`] verbatim. Negated HERE, not
///   by the caller: passing an already-negative `b` would reward mismatches. Same `i8` range.
/// - `mat`: the 25-entry matrix, written row-major and in full. Its prior contents are irrelevant
///   (no clearing needed), and it is the only thing this function mutates.
///
/// Free function rather than a method so [`Default`] can call it before a `MemOpt` exists.
fn fill_scmat(a: i32, b: i32, mat: &mut [i8; 25]) {
    // Reminder: `a` is the match reward (added), `b` the mismatch penalty (positive here, negated
    // into the matrix). `cell` walks the 25 entries row-major.
    let mut cell = 0usize;
    // Invariant at the top of each outer iteration: `cell == read_base * 5`, i.e. it points at the
    // first column of the row for this base, and every earlier row is fully written. The inner loop
    // fills columns 0..=3 and the statement after it fills column 4, so each pass advances `cell` by
    // exactly 5 and the single counter stays in step with the (row, column) pair without any
    // multiplication.
    for read_base in 0..4 {
        // Columns 0..=3 of this row: this base scored against each of the four real bases.
        for ref_base in 0..4 {
            mat[cell] = if read_base == ref_base {
                a as i8
            } else {
                -(b as i8)
            };
            cell += 1;
        }
        // Last column of each real-base row: that base against N.
        mat[cell] = AMBIGUOUS_BASE_SCORE;
        cell += 1;
    }
    // Row 4: N against anything, including N against N. Fixed at -1 rather than -b, so an N never
    // costs as much as a real mismatch: the base is unknown, not known-different.
    // `cell` is 20 on entry here (four rows of five already written) and 25, the full matrix, on
    // exit. The loop count is the row width, hence ALPHABET_SIZE rather than a literal 5.
    for _ in 0..ALPHABET_SIZE {
        mat[cell] = AMBIGUOUS_BASE_SCORE;
        cell += 1;
    }
}

/// Score for any pair involving N, in bwa's `bwa_fill_scmat`: a flat -1, independent of `-A`/`-B`.
///
/// A small constant penalty, not `-b`: an N is a base the sequencer could not call, so it is
/// unknown rather than known-different, and charging it a full mismatch would push reads with a
/// short N-run below `-T` for no evidential reason. Changing it silently shifts `AS:i:` on every
/// read containing an N and so breaks byte-parity with bwa-mem2; it is a hardcoded literal in the
/// C too, not a tunable.
const AMBIGUOUS_BASE_SCORE: i8 = -1;

/// A, C, G, T plus N: the matrix is [`ALPHABET_SIZE`] x [`ALPHABET_SIZE`] = 25 entries.
///
/// Fixed by the nt4 encoding in [`crate::dna`], not a tuning parameter. It cannot be changed
/// alone: [`MemOpt::mat`] is declared `[i8; 25]`, and the Smith-Waterman kernels index the matrix
/// with a hardcoded stride of 5.
const ALPHABET_SIZE: usize = 5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_bwa_mem2() {
        let o = MemOpt::default();
        assert_eq!((o.a, o.b), (1, 4));
        assert_eq!((o.o_del, o.e_del), (6, 1));
        assert_eq!(o.min_seed_len, 19);
        assert_eq!(o.max_occ, 500);
        assert_eq!(o.chunk_size, 10_000_000);
        assert_eq!(o.max_mem_intv, 20);
        // scoring matrix: diagonal = a, off-diagonal = -b, N row/col = -1
        assert_eq!(o.mat[0], 1); // A/A
        assert_eq!(o.mat[1], -4); // A/C
        assert_eq!(o.mat[4], -1); // A/N
        assert_eq!(o.mat[24], -1); // N/N
    }
}
