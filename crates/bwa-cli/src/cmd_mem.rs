//! `bwa-mem4 mem` subcommand: the command-line front end of the aligner.
//!
//! Ports `main_mem` from `reference/bwa-mem2/src/fastmap.cpp`. Its job is argv in, SAM bytes out:
//! parse options into [`bwa_core::MemOpt`], load the index, then stream batches of reads through
//! seed -> chain -> extend -> best region -> `reg2aln` (exact CIGAR + NM/MD), and format each
//! surviving region as a SAM record. The alignment mathematics all live in other crates
//! (`bwa-index`, `bwa-mem`, `bwa-chain`, `bwa-neon`); what lives here is option handling, batching,
//! threading, and record emission.
//!
//! # Reading order
//!
//! 1. [`MemArgs`]: the option surface, one clap field per bwa flag.
//! 2. [`parse_int_pair`] and [`parse_insert_size`]: bwa's two idiosyncratic argument syntaxes.
//! 3. [`build_opt`]: argv -> `MemOpt`, in three phases whose ORDER is load-bearing.
//! 4. [`Output`] and [`run_pipeline`]: where bytes go and how I/O is overlapped with compute.
//! 5. [`run`]: the single-end driver, and [`run_pe`] the paired-end one.
//! 6. [`finish_se`] and [`write_aln_se`]: which alignments become records, and the exact bytes.
//!
//! # Rust mechanics used in this file
//!
//! This is where the program's concurrency lives, so it is the densest file in the tree for
//! language features. Three ideas carry most of it.
//!
//! **Scoped threads.** `std::thread::scope` starts threads that are GUARANTEED to have finished
//! before the scope's closing brace. That guarantee is what lets those threads borrow local
//! variables directly: the compiler can see the data outlives them. Without it, everything shared
//! with a thread would have to be reference-counted and heap-allocated first.
//!
//! **Bounded channels.** A channel is a queue between threads. `sync_channel(n)` bounds it at `n`
//! items, so a producer that outruns its consumer BLOCKS rather than growing the queue without
//! limit. That bound is what caps how many read batches and how many finished SAM buffers exist at
//! once, and therefore caps peak memory on a 30x whole-genome run. The bound is in BATCHES, and a
//! batch is `-K` bases which itself defaults to 10M per thread, so a depth of one is already double
//! buffering and anything deeper is measured in hundreds of megabytes: see [`queue_depths`].
//!
//! **Indexed parallel iteration.** `rayon`'s `.par_iter()` spreads a loop across the thread pool.
//! The crucial property for this project is that it is INDEXED: results come back in input order,
//! not completion order. That is why parallelism here cannot perturb the emitted bytes, and it is
//! the reason `-t` does not appear anywhere in the output.
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `std::thread::scope(\|scope\| ...)` | opens a region in which threads may be started and are all joined before it ends. |
//! | `scope.spawn(move \|\| ...)` | starts one such thread running that closure. |
//! | `move \|\|` | a closure that TAKES OWNERSHIP of what it mentions rather than borrowing it. Required when handing work to another thread, since a borrow could otherwise outlive its owner. |
//! | `sync_channel::<T>(n)` | a queue of at most `n` values of type `T`, with a sender and a receiver half. |
//! | `SyncSender<T>` / receiver | the two ends. Sending on a full queue waits; receiving on an empty one waits; the receiver ends cleanly when every sender is gone. |
//! | `impl FnOnce(...) -> R + Send` | a parameter that is a callable, usable once, and safe to move to another thread. It is how [`run_pipeline`] takes the reader's body from its caller. |
//! | `.par_iter()` / `.into_par_iter()` | parallel walks over a collection, borrowing or consuming it. Order-preserving. |
//! | `.zip(...)` | walks two collections in step, pairing their elements. Used to carry a read, its encoded bases and its regions together through one parallel pass. |
//! | `Box<dyn Write + Send>` | an output sink whose concrete type is decided at run time (stdout, a file, a compressor) and which can be moved to the writer thread. `dyn` means the choice is resolved through a pointer, which is fine here: it is one indirect call per buffer, not per read. |
//! | `enum Output` with `impl Write` | the several output shapes gathered into one type that itself behaves as a sink, so the pipeline needs to know only "somewhere to write". |
//! | `#[derive(Args)]` | generates the option parser from the struct below. |
//! | `#[arg(short = 'k')]` | binds a field to a single-letter flag. The letters are bwa's, not clap's defaults, which is why `-h` and `-V` are explicitly disabled: bwa uses them for XA hits and the XR tag. |
//! | `anyhow::Result<T>` | a `Result` whose error can be any failure carrying context. Used at this layer, while the library crates use their own precise `bwa_core::Error`. |
//! | `?` | propagate a failure to the caller. |
//!
//! # Glossary
//!
//! Short names below mirror the C on purpose, because the correspondence is what makes this file
//! reviewable against `fastmap.cpp`. In plain language:
//!
//! | Name | Plain language |
//! |------|----------------|
//! | `fm` | the FM index: the data structure that answers "where does this exact string occur?" |
//! | `bns` | the contig dictionary plus the 2-bit packed reference (names, lengths, offsets) |
//! | `opt` | the [`bwa_core::MemOpt`] bundle of every tunable; see that module's own glossary |
//! | `l_pac` | length of the packed reference: every contig concatenated, forward then reverse strand |
//! | `rid` | reference (contig) id, an index into `bns.contigs`; printed as SAM's RNAME |
//! | seed | a stretch matching the reference exactly, the starting point for an alignment |
//! | chain | a set of seeds in a consistent order and roughly the right spacing to be one alignment |
//! | region (`MemAlnReg`) | a candidate alignment: a scored reference interval, before CIGAR |
//! | extension | banded Smith-Waterman outwards from a chain, turning it into a real alignment |
//! | band (`w`) | the diagonal stripe of the DP matrix actually computed; caps the indel length findable |
//! | `a`, `b` | match reward and mismatch penalty (see [`bwa_core::opt`]) |
//! | `o_del`/`e_del`, `o_ins`/`e_ins` | gap open and gap extend, for deletions and insertions |
//! | `zdrop` | abandon an extension once its score falls this far below its own best |
//! | `MEM_F_*` | the behaviour bits in `opt.flag`, e.g. `MEM_F_ALL` for `-a` |
//! | mate rescue | when one mate aligned and the other did not, search the window the insert size implies |
//! | `pes` | the four per-orientation insert-size distributions (see [`parse_insert_size`]) |
//!
//! SAM columns, for the emission functions: 1 QNAME (read name), 2 FLAG (bitfield), 3 RNAME
//! (contig), 4 POS (1-based), 5 MAPQ (confidence), 6 CIGAR (alignment ops), 7-9 RNEXT/PNEXT/TLEN
//! (the mate's whereabouts), 10 SEQ (bases), 11 QUAL (per-base qualities), then optional tags.

use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use clap::Args;
use rayon::prelude::*;
#[cfg(feature = "hts-output")]
use rust_htslib::bam;
// The raw htslib C bindings, re-exported by rust-htslib (so this is not an extra dependency). Used
// only by `HtsWriter`, which needs an ordering its safe `bam::Writer` cannot express; see there.
#[cfg(feature = "hts-output")]
use rust_htslib::htslib;

use bwa_core::opt::flags;
use bwa_core::{dna, MemOpt};
use bwa_index::{BntSeq, FmIndex};
use bwa_io::{sam, FastqReader, InterleavedFastqReader, PairedFastqReader, Record, SqRecord};
use bwa_mem::{
    align_reads_batched, alt::mem_gen_alt, batch_mate_rescue, cigar::cigar_string_which,
    cigar::MemAln, cigar_string, mem_approx_mapq_se, mem_mark_primary_se, mem_pestat, mem_sam_pe,
    mem_sort_dedup_patch, reg2aln, MemAlnReg, PairRescueData, PeStat,
};
use bwa_neon::NeonBackend;

use crate::stage_time::{self, Stage};

// ---- SAM FLAG bits set by this file (SAM spec section 1.4). The pairing bits 0x1/0x2/0x40/0x80
//      belong to the paired-end path in `bwa-mem` and never appear here ----

/// 0x10: the read aligned to the reverse strand, so SEQ is written reverse-complemented.
const FLAG_REVERSE: u32 = 0x10;
/// 0x100: secondary alignment, an alternative placement shadowed by a better one.
const FLAG_SECONDARY: u32 = 0x100;
/// 0x800: supplementary alignment, another piece of a split (chimeric) read.
const FLAG_SUPPLEMENTARY: u32 = 0x800;
/// NOT a SAM bit. bwa's internal marker for "supplementary, but `-M` asked for it to be reported as
/// secondary". It is folded into 0x100 and masked off before the FLAG is printed; see
/// [`write_aln_se`].
const FLAG_SUPP_REPORTED_AS_SECONDARY: u32 = 0x10000;
/// Mask keeping only the 16 real SAM flag bits, dropping the pseudo-bit above.
const FLAG_SAM_BITS: u32 = 0xffff;

// ---- CIGAR operation codes. Careful: bwa uses its OWN 5-op table `MIDSH` (`bwamem.h:49`), NOT the
//      SAM/BAM spec's `MIDNSH`. So here 3 = S and 4 = H, where the spec would say 3 = N and 4 = S ----

/// Soft clip in bwa's `MIDSH` numbering (the SAM spec would call 3 `N`).
const BWA_CIGAR_OP_SOFT_CLIP: u32 = 3;
/// Hard clip in bwa's `MIDSH` numbering (the SAM spec would call 4 `S`).
const BWA_CIGAR_OP_HARD_CLIP: u32 = 4;
/// A cigar entry is packed as `oplen << 4 | op`, so the op is the low nibble.
const CIGAR_OP_MASK: u32 = 0xf;
/// ... and the length lives in the remaining high bits.
const CIGAR_LEN_SHIFT: u32 = 4;

/// `bwa-mem2 mem`'s option set. Short flags, defaults and semantics mirror `main_mem`
/// (`reference/bwa-mem2/src/fastmap.cpp`) so the two CLIs are interchangeable.
///
/// Scalar options are `Option<T>` on purpose: that is bwa's `opt0` "was it given explicitly?"
/// record, which [`build_opt`] needs to reproduce `update_a`'s `-A` rescaling.
///
/// clap's automatic `-h`/`-V` are disabled: bwa uses `-h` for XA hits and `-V` for the XR tag.
///
/// # Reading this struct
///
/// The `///` line on each field is clap's help text and is kept verbatim from bwa's own usage
/// output (`fastmap.cpp:586-620`) so `--help` reads the same. The `//` lines above each field are
/// the explanation: meaning, units, default, range, the C variable it lands in, and interactions.
/// Full per-parameter semantics live on the corresponding [`bwa_core::MemOpt`] field; this struct is
/// only the transport from argv into it.
///
/// # Two rules that govern the whole struct
///
/// 1. `Option<T>` is bwa's `opt0` in disguise. The C keeps a second, zeroed `mem_opt_t opt0` and
///    sets `opt0.<field> = 1` whenever the user supplies that option (`fastmap.cpp:656-745`), purely
///    so `update_a` can later tell "user asked for 6" from "default happens to be 6". `None` here
///    means the same thing as `opt0.<field> == 0`.
/// 2. Order of application is fixed: parse everything, then `update_a`, then fill the scoring
///    matrix. See [`build_opt`].
///
/// # Scope note
///
/// bwa's getopt string is `"51qpaMCSPVYjk:c:v:s:r:t:R:A:B:O:E:U:w:L:d:T:Q:D:m:I:N:W:x:G:h:y:K:X:H:o:f:"`
/// (`fastmap.cpp:660`). Every letter in it is now accepted. Three deserve a note:
/// - `-x STR` (preset modes `pacbio`/`pbref`/`ont2d`/`intractg`): rewrites up to nine options at
///   once (`fastmap.cpp:820-855`) and takes a *different* branch to `update_a`, so a preset
///   suppresses the `-A` rescaling. Implemented in PHASE 2 of [`build_opt`].
/// - `-1` (disable multithreaded I/O): accepted and INERT. This port's reader and writer are
///   separate threads by construction, and the flag cannot affect SAM bytes in either
///   implementation.
/// - `-f FILE`: accepted as a short alias of `-o`, which is exactly what it is in bwa
///   (`fastmap.cpp:674`, one shared branch).
///
/// Two accepted options are inert and marked NO-OP at their fields: `-j` and `-v`. Both are inert
/// for reasons that cannot affect SAM bytes, explained there.
// `Default` is for embedders, not for the CLI: clap fills every field from the command line, but a
// caller using this crate as a library has no command line and would otherwise have to name all
// forty-odd fields to set two of them. The only field carrying a clap default is `-t`
// (`default_value_t = 1`), and `Default` gives 0 there rather than 1; that is harmless because
// `run` clamps with `args.threads.max(1)`, so both routes start one thread when nothing asks for
// more. Every other field's clap behaviour when absent IS its `Default`, since they are `Option`
// or `bool`.
#[derive(Args, Default)]
#[command(disable_help_flag = true, disable_version_flag = true)]
pub struct MemArgs {
    // `-t INT` -> `opt->n_threads` (`fastmap.cpp:672`). Default 1, clamped to >= 1 by both
    // implementations. Purely a speed knob: unlike most aligners, nothing about the output depends
    // on it here, because reads are numbered globally and emitted in input order.
    /// Number of threads [1]. Output order and global read ids are independent of it, so
    /// byte-identity holds at any `-t` once `-K` fixes the batch boundaries.
    #[arg(short = 't', default_value_t = 1)]
    pub threads: i32,
    // `-K INT` -> the batch size in input bases. bwa stores it in a local `fixed_chunk_size`
    // (`fastmap.cpp:709`) rather than in `opt`, then folds it into `opt->chunk_size` at
    // `fastmap.cpp:966`. Default here (see `run`) is `chunk_size * threads` = 10M * `-t`, matching
    // `aux.task_size = opt->chunk_size * opt->n_threads` (`fastmap.cpp:964`).
    //
    // NOT a pure performance knob for paired-end input: `mem_pestat` estimates the insert-size
    // distribution once per batch, so moving a batch boundary can change MAPQ and pairing decisions
    // for the reads near it. Pin `-K` to make a run reproducible across `-t` values.
    /// Process INT input bases per batch (fixes batch boundaries for reproducibility).
    #[arg(short = 'K')]
    pub k_batch: Option<i64>,

    // ---- Algorithm options ----
    // `-k INT` -> `opt->min_seed_len` (`fastmap.cpp:662`). Bases. Default 19, useful range ~10..40.
    // Below ~15 seeds hit a mammalian genome so often that runtime explodes; above ~30 reads with
    // dense SNPs stop seeding at all. Interacts with `-r` (re-seed trigger is `min_seed_len * r`).
    /// Minimum seed length [19].
    #[arg(short = 'k')]
    pub min_seed_len: Option<i32>,
    // `-w INT` -> `opt->w` (`fastmap.cpp:665`). Bases. Default 100. Upper bound on the indel length
    // one banded Smith-Waterman extension can find. NOT rescaled by `-A`: it is a length.
    /// Band width for banded alignment [100].
    #[arg(short = 'w')]
    pub band_width: Option<i32>,
    // `-d INT` -> `opt->zdrop` (`fastmap.cpp:693`). Score units. Default 100. Cuts an extension once
    // it falls this far below its own best score off-diagonal, which is how a read spanning a
    // structural breakpoint becomes two records instead of one implausible CIGAR. RESCALED by `-A`.
    /// Off-diagonal X-dropoff (Z-drop) [100].
    #[arg(short = 'd')]
    pub zdrop: Option<i32>,
    // `-r FLOAT` -> `opt->split_factor` (`fastmap.cpp:696`). Dimensionless, default 1.5, sensible
    // range >= 1.0. Any SMEM longer than `-k * -r` gets re-seeded from its middle to expose shorter
    // seeds it masked. Note the C sets `opt0.split_factor = 1.` (a float) rather than 1.
    /// Look for internal seeds inside a seed longer than {-k} * FLOAT [1.5].
    #[arg(short = 'r')]
    pub split_factor: Option<f32>,
    // `-y INT` -> `opt->max_mem_intv` (`fastmap.cpp:707`, read with `atol`, hence i64). Occurrence
    // count, default 20. Threshold for bwa's third seeding round, which recovers moderately
    // repetitive seeds the first two rounds dropped.
    /// Seed occurrence for the 3rd round of seeding [20].
    #[arg(short = 'y')]
    pub max_mem_intv: Option<i64>,
    // `-c INT` -> `opt->max_occ` (`fastmap.cpp:692`). Occurrence count, default 500. The repeat
    // guard: a seed landing in a satellite array can occur tens of thousands of times and extending
    // each is pure waste. Raising it buys repeat sensitivity at steep runtime cost.
    /// Skip seeds with more than INT occurrences [500].
    #[arg(short = 'c')]
    pub max_occ: Option<i32>,
    // `-D FLOAT` -> `opt->drop_ratio` (`fastmap.cpp:698`). Fraction in 0.0..=1.0, default 0.50.
    // Also reused at output time as the floor for emitting a secondary record under `-a` (see
    // `finish_se`), where the C compares in single precision.
    /// Drop chains shorter than FLOAT fraction of the longest overlapping chain [0.50].
    #[arg(short = 'D')]
    pub drop_ratio: Option<f32>,
    // `-W INT` -> `opt->min_chain_weight` (`fastmap.cpp:705`). Bases, default 0 (off). bwa's
    // long-read presets raise it to 20 (`pacbio`) or 40 (`ont2d`); this port has no `-x`, so it is
    // only ever what the user passes.
    /// Discard a chain if seeded bases are shorter than INT [0].
    #[arg(short = 'W')]
    pub min_chain_weight: Option<i32>,
    // `-m INT` -> `opt->max_matesw` (`fastmap.cpp:699`). Rounds, default 50. Paired-end only, and
    // moot under `-S`. Bounds the worst case on reads whose mate has many candidate positions.
    /// Perform at most INT rounds of mate rescue for each read [50].
    #[arg(short = 'm')]
    pub max_matesw: Option<i32>,
    // `-S` -> `MEM_F_NO_RESCUE` (`fastmap.cpp:687`). Boolean. Turns off the pass that Smith-Waterman
    // aligns an unmapped mate inside the reference window implied by the insert-size model. Changes
    // the output by design; it is roughly half of paired-end wall time here.
    /// Skip mate rescue.
    #[arg(short = 'S')]
    pub skip_mate_rescue: bool,
    // `-P` -> `MEM_F_NOPAIRING` (`fastmap.cpp:683`). Boolean. Each mate keeps its own best
    // single-end alignment; the pair is never scored jointly, so `-U` becomes irrelevant. Mate
    // rescue still runs unless `-S` is also given, which is why the two flags are separate.
    /// Skip pairing; mate rescue still performed unless -S is also given.
    #[arg(short = 'P')]
    pub skip_pairing: bool,
    // `-G INT` -> `opt->max_chain_gap` (`fastmap.cpp:701`). Bases, default 10000. Two seeds further
    // apart than this are never put in the same chain, so this bounds the deletion a single
    // alignment record can span. Absent from bwa's printed usage but fully parsed.
    /// Max chain gap [10000] (bwa's undocumented -G).
    #[arg(short = 'G')]
    pub max_chain_gap: Option<i32>,
    // `-N INT` -> `opt->max_chain_extend` (`fastmap.cpp:703`). Count, default 1<<30 (unlimited in
    // practice). Also absent from bwa's usage text.
    /// Max chain extension (bwa's undocumented -N).
    #[arg(short = 'N')]
    pub max_chain_extend: Option<i32>,
    // `-s INT` -> `opt->split_width` (`fastmap.cpp:700`). Occurrence count, default 10. Re-seeding
    // (see `-r`) only fires when the long SMEM occurs fewer than this many times.
    /// Re-seed occurrence threshold [10] (bwa's undocumented -s).
    #[arg(short = 's')]
    pub split_width: Option<i32>,
    // `-X FLOAT` -> `opt->mask_level` (`fastmap.cpp:711`). Fraction 0.0..=1.0, default 0.50. Two
    // hits are redundant when they overlap by more than this fraction of the shorter one.
    // Quirk worth knowing: the C does NOT set `opt0.mask_level`, so `-X` is invisible to
    // `update_a` and to the `-x` presets. It is a float, so it would never be rescaled anyway.
    /// Redundancy mask level [0.50] (bwa's undocumented -X).
    #[arg(short = 'X')]
    pub mask_level: Option<f32>,
    // `-Q INT` -> `opt->mapQ_coef_len` AND, derived, `opt->mapQ_coef_fac` (`fastmap.cpp:719-723`).
    // Bases, default 50. Setting it recomputes `fac = len > 0 ? log(len) : 0`; see `build_opt` for
    // the integer-truncation quirk that comes with the C's `int mapQ_coef_fac`.
    /// MAPQ coefficient length [50] (bwa's undocumented -Q).
    #[arg(short = 'Q')]
    pub mapq_coef_len: Option<i32>,

    // ---- Scoring options ----
    // `-A INT` -> `opt->a` (`fastmap.cpp:666`). Score units, default 1, positive.
    //
    // THE interacting option. Every other score is denominated in units of `a`, so bwa's `update_a`
    // (`fastmap.cpp:564-577`) multiplies each of `-T -d -B -O -E -L -U` by the new `a` UNLESS the
    // user set that option explicitly. `-A 2` therefore gives a uniformly doubled scoring system,
    // not a suddenly permissive aligner. `-A 2 -B 4` gives `b=4` (kept) with everything else
    // doubled. See `build_opt` for the reproduction, and `MemOpt::fill_scmat` for why the scoring
    // matrix must be filled after all of this and not during parsing.
    /// Score for a sequence match, which scales options -TdBOELU unless overridden [1].
    #[arg(short = 'A')]
    pub match_score: Option<i32>,
    // `-B INT` -> `opt->b` (`fastmap.cpp:667`). Score units, default 4, positive magnitude
    // (subtracted at use). The `-A`:`-B` ratio sets tolerated divergence: at 1:4 a mismatch costs
    // four matches. RESCALED by `-A` when not given.
    /// Penalty for a mismatch [4].
    #[arg(short = 'B')]
    pub mismatch: Option<i32>,
    // `-O INT[,INT]` -> `opt->o_del`, `opt->o_ins` (`fastmap.cpp:725-731`). Score units, default
    // 6,6, positive magnitudes. Charged ONCE per gap. One INT sets both; the pair is deletion then
    // insertion. Parsed as a String because of bwa's idiosyncratic pair syntax, see
    // `parse_int_pair`. RESCALED by `-A` when not given (both halves together: the C sets
    // `opt0.o_del = opt0.o_ins = 1` from the single option).
    /// Gap open penalties for deletions and insertions, INT[,INT] [6,6].
    #[arg(short = 'O')]
    pub gap_open: Option<String>,
    // `-E INT[,INT]` -> `opt->e_del`, `opt->e_ins` (`fastmap.cpp:732-738`). Score units, default
    // 1,1. Charged PER GAP BASE, so a length-k gap costs `o + e*k` (affine gap model). RESCALED by
    // `-A` when not given.
    /// Gap extension penalty, INT[,INT]; a gap of size k costs {-O} + {-E}*k [1,1].
    #[arg(short = 'E')]
    pub gap_extend: Option<String>,
    // `-L INT[,INT]` -> `opt->pen_clip5`, `opt->pen_clip3` (`fastmap.cpp:739-745`). Score units,
    // default 5,5. Biases towards end-to-end alignment by making clipping pay for itself. Per the
    // C's own note (`bwamem.h:80`) this is NOT deducted from the reported `AS:i:` score; it only
    // steers the local-vs-glocal choice. RESCALED by `-A` when not given.
    /// Penalty for 5'- and 3'-end clipping, INT[,INT] [5,5].
    #[arg(short = 'L')]
    pub clip_penalty: Option<String>,
    // `-U INT` -> `opt->pen_unpaired` (`fastmap.cpp:669`). Score units, default 17, paired-end only.
    // Charged when bwa reports a pair as unpaired rather than forcing it into the inferred insert
    // window; raising it biases towards "properly paired". Meaningless under `-P`. RESCALED by `-A`.
    /// Penalty for an unpaired read pair [17].
    #[arg(short = 'U')]
    pub pen_unpaired: Option<i32>,

    // ---- Input/output options ----
    // `-T INT` -> `opt->T` (`fastmap.cpp:668`). Score units, default 30. Output-only: it does not
    // change how alignments are found, only which are written. A read with no region reaching `-T`
    // is emitted as an unmapped FLAG 4 record, not omitted. RESCALED by `-A` when not given.
    /// Minimum score to output [30].
    #[arg(short = 'T')]
    pub min_score: Option<i32>,
    // `-h INT[,INT]` -> `opt->max_XA_hits`, `opt->max_XA_hits_alt` (`fastmap.cpp:712-718`). Counts,
    // default 5,200. If a read has at most this many hits scoring above `XA_drop_ratio` (0.80,
    // not settable) times the best, they are all listed in the `XA:Z` tag rather than discarded.
    // The ALT limit is far higher because an ALT haplotype is expected to collide with the primary
    // assembly repeatedly. Interaction: `-a` suppresses `XA:Z` entirely, making `-h` moot.
    // Note bwa uses `-h` for this, which is why clap's automatic `-h`/help is disabled above.
    /// If there are <INT hits with score >80% of the max score, output all in XA, INT[,INT] [5,200].
    #[arg(short = 'h')]
    pub xa_hits: Option<String>,
    // `-a` -> `MEM_F_ALL` (`fastmap.cpp:684`). Boolean. Emits shadowed hits as their own FLAG 0x100
    // secondary records (MAPQ 0, SEQ/QUAL `*`) instead of folding them into `XA:Z`. The two are
    // mutually exclusive by design: the same hits must not be reported twice.
    /// Output all alignments for SE or unpaired PE.
    #[arg(short = 'a')]
    pub output_all: bool,
    // `-M` -> `MEM_F_NO_MULTI` (`fastmap.cpp:686`). Boolean. Compatibility switch for tools that
    // predate FLAG 0x800: a supplementary record is reported as secondary 0x100 instead. See
    // `write_aln_se` for the 0x10000 intermediate bit bwa uses to implement this cleanly.
    /// Mark shorter split hits as secondary.
    #[arg(short = 'M')]
    pub mark_secondary: bool,
    // `-Y` -> `MEM_F_SOFTCLIP` (`fastmap.cpp:688`). Boolean. Supplementary records normally
    // hard-clip (`H`) so the read's bases appear exactly once across all its records; `-Y`
    // soft-clips (`S`) so every record carries the full SEQ. Downstream tools differ on which they
    // want; the cost of `-Y` is duplicated sequence in the file.
    /// Use soft clipping for supplementary alignments.
    #[arg(short = 'Y')]
    pub soft_clip_supp: bool,
    // `-C` -> `aux.copy_comment` (`fastmap.cpp:708`), NOT a MEM_F_ flag. Boolean. Appends the FASTQ
    // header's post-whitespace comment to the very end of every SAM record, after all tags. Held in
    // process-wide state here to mirror the C, see `bwa_core::rg`.
    /// Append FASTA/FASTQ comment to SAM output.
    #[arg(short = 'C')]
    pub copy_comment: bool,
    // `-V` -> `MEM_F_REF_HDR` (`fastmap.cpp:689`). Boolean. bwa uses `-V` for this, which is why
    // clap's automatic `-V`/version is disabled above.
    /// Output the reference FASTA header in the XR tag.
    #[arg(short = 'V')]
    pub ref_hdr: bool,
    // `-5` -> `MEM_F_PRIMARY5 | MEM_F_KEEP_SUPP_MAPQ` (`fastmap.cpp:690`). Boolean. For Hi-C and
    // other chimeric libraries, where the 5'-most segment is the meaningful one regardless of which
    // scored best. bwa ALWAYS sets KEEP_SUPP_MAPQ with it (the C carries that exact comment), so
    // `-5` implies `-q`.
    /// For split alignment, take the alignment with the smallest coordinate as primary.
    #[arg(short = '5')]
    pub primary5: bool,
    // `-q` -> `MEM_F_KEEP_SUPP_MAPQ` (`fastmap.cpp:691`). Boolean. Without it, a later record's
    // MAPQ is capped at the first record's, on the reasoning that a supplementary alignment cannot
    // be more confident than the primary it was split from. Implied by `-5`.
    /// Don't modify the MAPQ of supplementary alignments.
    #[arg(short = 'q')]
    pub keep_supp_mapq: bool,
    // `-j` -> `ignore_alt = 1` (`fastmap.cpp:694`), a local in `main_mem`, not an opt field.
    //
    // NO-OP in this port, and harmlessly so: `bwa_index::BntSeq`'s `Contig` has no `is_alt` field
    // and no `.alt` file is ever read, so ALT contigs are already treated as primary-assembly
    // sequence. Accepting the flag keeps the CLIs interchangeable; it cannot change output because
    // the behaviour it disables is not implemented. (Consequently `max_xa_hits_alt` is also never
    // the limit used.)
    /// Treat ALT contigs as part of the primary assembly (ignore the <idxbase>.alt file).
    #[arg(short = 'j')]
    pub ignore_alt: bool,
    // `-p` -> `MEM_F_PE | MEM_F_SMARTPE` (`fastmap.cpp:685`). Boolean. Both mates come interleaved
    // from the single input file. See `InterleavedFastqReader` for the one place this port
    // deliberately refuses rather than mimic bwa (mixed paired/singleton input).
    /// Smart pairing: mates are interleaved in the single input file (ignores in2.fq).
    #[arg(short = 'p')]
    pub smart_pairing: bool,
    // `-R STR` -> `bwa_set_rg(optarg)` (`fastmap.cpp:746`). The literal `@RG` header line, with
    // backslash escapes (`\t` etc.) expanded. Must start with `@RG` and contain a `\tID:` field of
    // at most 255 characters or bwa exits. The parsed ID is then stamped as `RG:Z:<id>` on EVERY
    // record, unmapped ones included. Interaction with `-H`: the `@RG` line is appended AFTER the
    // `-H` lines in the header, because the C only calls `bwa_insert_header(rg_line, hdr_line)`
    // once the getopt loop has finished. See `run`.
    /// Read group header line, such as '@RG\tID:foo\tSM:bar'.
    #[arg(short = 'R')]
    pub read_group: Option<String>,
    // `-H STR|FILE` -> `hdr_line` via `bwa_insert_header` (`fastmap.cpp:757-775`). If the argument
    // starts with `@` it is one literal header line; otherwise it is a path whose `@`-prefixed
    // lines are inserted. Interaction with the generated header: supplying any `@SQ` line here
    // SUPPRESSES all the generated `@SQ` lines (`bwa_print_sam_hdr`, `bwa.cpp:523`).
    /// Insert STR into the header if it starts with @, or insert the lines of FILE.
    #[arg(short = 'H')]
    pub header_insert: Option<String>,
    // `-I FLOAT[,FLOAT[,INT[,INT]]]` -> `pes[1]` and `aux.pes0` (`fastmap.cpp:777-793`).
    // mean,stddev,max,min of the insert size, in bases. Fixes the distribution instead of inferring
    // it per batch, so it also removes `-K`'s influence on paired-end output. Only the FR
    // orientation is filled; the other three stay `failed`. See `parse_insert_size` for the
    // defaulting rules (std = 10% of mean, bounds = mean +/- 4 std).
    /// Insert size distribution: FLOAT[,FLOAT[,INT[,INT]]] = mean,std,max,min.
    #[arg(short = 'I')]
    pub insert_size: Option<String>,
    // `-v INT` -> the C's global `bwa_verbose` (`fastmap.cpp:694`), not an opt field.
    //
    // NO-OP in this port: nothing reads it. It only ever gated stderr chatter in bwa (and one
    // warning in `bwa_print_sam_hdr` about an `-H` @SQ count mismatch), never SAM bytes, so
    // accepting and ignoring it cannot change output.
    /// Verbose level: 1=error, 2=warning, 3=message, 4+=debugging [3].
    #[arg(short = 'v')]
    pub verbose: Option<i32>,

    // Positional 1. The prefix passed to `bwa-mem4 index`, i.e. the FASTA path itself. The index
    // side files (`.bwt.2bit.64`, `.ann`, `.amb`, `.pac`, ...) are found by appending suffixes.
    /// Index prefix: the FASTA path that was indexed.
    pub index_prefix: PathBuf,
    // Positional 2. FASTQ or FASTA, optionally gzipped (needletail sniffs the compression).
    /// Reads in FASTQ (R1, or the only file for single-end).
    pub reads: PathBuf,
    // Positional 3, optional. Its mere presence is what switches on paired-end mode, mirroring the
    // C, which ORs in `MEM_F_PE` when it sees a second input file (`fastmap.cpp:953`). Ignored
    // under `-p`, where both mates come from `reads`.
    /// Optional mate reads (R2): triggers paired-end mode.
    pub reads2: Option<PathBuf>,
    // `-o FILE` -> `aux.fp = fopen(optarg, "w")` (`fastmap.cpp:674`). bwa accepts `-f` as an exact
    // alias (same branch); this port does not, see the scope note above. The BGZF/BAM/CRAM
    // behaviour below is an addition, not a bwa feature: bwa always writes plain SAM. The records
    // are the same bytes in every case, since BAM and CRAM transcode the formatted SAM text.
    /// Write SAM to PATH instead of stdout. A `.gz`/`.bgz` suffix selects BGZF (block-gzip) output,
    /// compressed in parallel on `-t` worker threads (readable by samtools/bgzip/tabix); `.bam` and
    /// `.cram` select binary BAM/CRAM via htslib (`.cram` also needs `--reference`).
    #[arg(short = 'o', long, short_alias = 'f')]
    pub output: Option<PathBuf>,
    // `-x STR` -> `mode` (`fastmap.cpp:665`), applied AFTER the whole getopt loop
    // (`fastmap.cpp:818-859`). Not a spelling: it rewrites up to nine options at once, each only
    // where the user left the default, and it takes a DIFFERENT branch to `update_a`, so giving
    // `-x` suppresses the `-A` rescaling entirely. See PHASE 2 in `build_opt`.
    //
    // bwa-mem2 prints a warning for any mode ("doesn't work well with long reads or contigs; please
    // use minimap2 instead") and this port prints it too: it goes to stderr, so it cannot touch SAM
    // bytes, and a user reaching for `-x pacbio` deserves the same steer bwa gives them.
    /// Read type preset: `pacbio`, `pbref`, `ont2d` or `intractg`. Rewrites several scoring and
    /// seeding options at once, each only where it was not given explicitly.
    #[arg(short = 'x')]
    pub read_type: Option<String>,
    // `-1` -> `no_mt_io` (`fastmap.cpp:664`), which makes bwa-mem2 read and write on the worker
    // thread instead of a dedicated one. ACCEPTED AND INERT here, like `-j` and `-v`: this port's
    // reader and writer are separate threads by construction and there is no single-threaded I/O
    // path to fall back to. Accepting it matters anyway, because a script that passes `-1` to
    // bwa-mem2 must not die on an unknown argument here, and the flag cannot affect SAM bytes in
    // either implementation.
    /// NO-OP, accepted for command-line compatibility: bwa's "disable multithreaded I/O".
    #[arg(short = '1')]
    pub no_mt_io: bool,
    // Long-only and not a bwa flag: bwa has no CRAM output, so there is no short letter to match.
    // Only CRAM reads it. It is the FASTA the CRAM decoder will need to reconstruct SEQ, so it must
    // be the same sequences as the index prefix; htslib loads it through its `.fai`.
    /// Reference FASTA for CRAM output (`-o out.cram`). A `.fai` beside it is used if present;
    /// without one htslib scans the FASTA to index it in memory, which is slower but works.
    #[arg(long = "reference", value_name = "FASTA")]
    pub reference: Option<PathBuf>,
    // Long-only, because `-h` is bwa's XA-hits option. bwa itself has no help flag: any
    // unrecognised option falls through to `return 1` and prints usage (`fastmap.cpp:795`).
    /// Print help.
    #[arg(long = "help", action = clap::ArgAction::Help)]
    pub help: Option<bool>,
}

/// Parse bwa's `INT[,INT]` pair syntax. bwa reads the first integer with `strtol`, then takes a
/// second one only when the next byte is punctuation *followed by a digit*
/// (`if (*p != 0 && ispunct(*p) && isdigit(p[1]))`); anything else leaves the second value unset.
///
/// Used by `-O`, `-E`, `-L` and `-h`. The odd rule is worth reproducing exactly: it means `6,6`,
/// `6:6`, `6/6` and `6-6` are ALL accepted as pairs (any punctuation separates), while `6,` and
/// `6x6` silently parse as the single value 6 rather than erroring. A user who typos `-O 6.5`
/// gets `o_del=6, o_ins=5` from both implementations.
///
/// Returns `(first, Some(second))` or `(first, None)`; the caller decides that `None` means "use
/// `first` for both", which is what the C's `opt->o_del = opt->o_ins = strtol(...)` does up front.
/// Errors only when no leading integer is present at all.
///
/// # Parameters
///
/// - `s`: the raw option argument exactly as the user typed it, straight out of clap (never `None`,
///   since the caller only reaches here for options that were supplied). Any trailing junk after
///   the recognised digits is ignored rather than rejected, matching `strtol`.
///
/// # Returns
///
/// `(first, second)` in score units for `-O`/`-E`/`-L` and in hit counts for `-h`; this function
/// does not know or care which. `Err` only for "no leading integer at all"; overflow of `i32` also
/// errors, via `parse`.
fn parse_int_pair(s: &str) -> anyhow::Result<(i32, Option<i32>)> {
    // Byte view of the argument: the scan is ASCII-only (digits, sign, punctuation), so indexing
    // bytes cannot split a multi-byte char in any string that parses.
    let bytes = s.as_bytes();
    // Cursor into `bytes`. Invariant: everything before `pos` has already been consumed by the
    // scan, and `pos` never moves backwards.
    let mut pos = 0;

    // ---- First integer: optional sign, then digits ----
    if pos < bytes.len() && (bytes[pos] == b'-' || bytes[pos] == b'+') {
        pos += 1;
    }
    // Where the first number's digits begin, i.e. `pos` past any sign. Compared against `pos` after
    // the digit loop purely to detect "sign but no digits", which is the one error case.
    let digits_start = pos;
    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
        pos += 1;
    }
    if pos == digits_start {
        anyhow::bail!("expected an integer in '{s}'");
    }
    // The leading value: `-O`/`-E`'s deletion half, `-L`'s 5' half, `-h`'s primary-hit limit.
    let first: i32 = s[..pos].parse()?;

    // ---- Second integer, only when the next byte is punctuation FOLLOWED BY a digit. Any
    //      punctuation separates, which is why `6,6`, `6:6` and `6.5` are all read as pairs ----
    if pos + 1 < bytes.len() && bytes[pos].is_ascii_punctuation() && bytes[pos + 1].is_ascii_digit()
    {
        // First byte of the second number: one past the separating punctuation, and already known
        // to be a digit (the guard above tested it), so no sign is accepted here. That asymmetry is
        // bwa's, not ours: `-O 6,-2` parses as `(6, None)`, the `,-` failing the digit test.
        let second_start = pos + 1;
        // One past the second number's last digit; grows in the loop below.
        let mut second_end = second_start;
        while second_end < bytes.len() && bytes[second_end].is_ascii_digit() {
            second_end += 1;
        }
        return Ok((first, Some(s[second_start..second_end].parse()?)));
    }
    Ok((first, None))
}

/// Parse `-I FLOAT[,FLOAT[,INT[,INT]]]` into the four per-orientation insert-size stats.
///
/// Port of `main_mem`'s `-I` branch: all four orientations start `failed`, then only `pes[1]` (FR)
/// is filled. std defaults to 10% of the mean, high/low to mean +/- 4 std, each rounded with the
/// C's `(int)(x + .499)`, and low is clamped to >= 1. Each further field is read only when the next
/// byte is punctuation followed by a digit, exactly as the C's `strtod` walk does.
///
/// # Why only `pes[1]`
///
/// The four slots are the four relative orientations of a read pair: index 0 = FF, 1 = FR, 2 = RF,
/// 3 = RR. Standard Illumina paired-end libraries are FR (mates point towards each other), so bwa
/// only lets `-I` describe that one and leaves the others `failed`, meaning "no evidence, do not
/// pair in this orientation". A `failed` orientation is not an error, it is the normal state for
/// three of the four.
///
/// # Units and returned invariants
///
/// `avg`/`std` are bases (f64); `low`/`high` are the inclusive bases bounds inside which a pair
/// counts as properly paired (i32, `low >= 1`). Note `.499` not `.5`: it is bwa's rounding and it
/// differs at exact halves, which is enough to move a boundary pair's FLAG.
///
/// # Parameters
///
/// - `s`: the raw `-I` argument as typed. One to four numbers, in the fixed order mean, std, max,
///   min, separated by any punctuation. Extra numbers past the fourth are parsed and then ignored,
///   as in the C.
///
/// # Returns
///
/// All four orientation slots, of which only `pes[ORIENTATION_FR]` is usable; the other three carry
/// `failed = true`. Errors when `s` contains no leading number at all, or when a number does not
/// parse as `f64`.
fn parse_insert_size(s: &str) -> anyhow::Result<[PeStat; 4]> {
    // ---- Walk successive numbers, using bwa's "punctuation then digit" continuation rule ----
    // The numbers as typed, in order: [0] mean, [1] std, [2] max, [3] min, all in bases. Length
    // 1..=4 in practice and read positionally below, so a missing field can only be a trailing one.
    let mut values: Vec<f64> = Vec::new();
    // Byte view for the ASCII scan, and the cursor into it. Invariant at the top of each loop turn:
    // `pos` sits at the first byte of the next number, everything before it is consumed.
    let bytes = s.as_bytes();
    let mut pos = 0;
    loop {
        // Start of the number being read this turn; equal to `pos` after the loop when there was no
        // number left, which is the loop's exit test.
        let number_start = pos;
        if pos < bytes.len() && (bytes[pos] == b'-' || bytes[pos] == b'+') {
            pos += 1;
        }
        while pos < bytes.len() && (bytes[pos].is_ascii_digit() || bytes[pos] == b'.') {
            pos += 1;
        }
        if pos == number_start {
            break;
        }
        values.push(s[number_start..pos].parse()?);
        if pos + 1 < bytes.len()
            && bytes[pos].is_ascii_punctuation()
            && bytes[pos + 1].is_ascii_digit()
        {
            pos += 1;
        } else {
            break;
        }
    }
    if values.is_empty() {
        anyhow::bail!("-I: expected at least the mean insert size");
    }

    // ---- All four orientations start `failed`; only FR is describable through `-I` ----
    // The result under construction, indexed 0 = FF, 1 = FR, 2 = RF, 3 = RR. `failed` is set on all
    // four first so that the three `-I` cannot describe stay "no evidence, never pair this way".
    let mut pes = [PeStat::default(); 4];
    for orientation in pes.iter_mut() {
        orientation.failed = true;
    }
    // Borrow of the one slot `-I` fills. Held for the rest of the function so the field writes read
    // like the C's `pes[1].avg = ...`.
    let fr = &mut pes[ORIENTATION_FR];

    // ---- Mean, then std (defaulting to a tenth of the mean), then the derived bounds ----
    fr.failed = false;
    fr.avg = values[0];
    fr.std = if values.len() > 1 {
        values[1]
    } else {
        fr.avg * DEFAULT_STD_AS_FRACTION_OF_MEAN
    };
    fr.high = (fr.avg + INSERT_BOUND_STD_MULTIPLE * fr.std + C_ROUND_BIAS) as i32;
    fr.low = (fr.avg - INSERT_BOUND_STD_MULTIPLE * fr.std + C_ROUND_BIAS) as i32;
    if fr.low < 1 {
        fr.low = 1;
    }

    // ---- Explicit max/min override the derived bounds ----
    if values.len() > 2 {
        fr.high = (values[2] + C_ROUND_BIAS) as i32;
    }
    if values.len() > 3 {
        fr.low = (values[3] + C_ROUND_BIAS) as i32;
    }
    Ok(pes)
}

/// Index of the FR (forward-reverse) orientation in the four-slot `pes` array: 0 = FF, 1 = FR,
/// 2 = RF, 3 = RR. FR is standard Illumina paired-end (the mates point towards each other) and is
/// the only orientation `-I` can describe.
const ORIENTATION_FR: usize = 1;

/// bwa's rounding constant. It is `.499`, NOT `.5`, so `(int)(x + .499)` differs from true
/// round-half-up at exact halves. Reproduced verbatim: the difference is enough to move a boundary
/// pair's FLAG.
const C_ROUND_BIAS: f64 = 0.499;

/// When `-I` gives only a mean, bwa takes the standard deviation to be a tenth of it.
const DEFAULT_STD_AS_FRACTION_OF_MEAN: f64 = 0.1;

/// The derived properly-paired window is mean +/- this many standard deviations.
const INSERT_BOUND_STD_MULTIPLE: f64 = 4.0;

/// Build [`MemOpt`] from the CLI, reproducing `main_mem`'s order: apply every explicitly given
/// option, then `update_a` (rescale the penalties the user did *not* set when `-A` changed the match
/// score), then `bwa_fill_scmat` last.
///
/// # The three phases, and why the order is not negotiable
///
/// 1. **Apply.** Every explicitly-given option overwrites its default. Nothing is derived yet.
/// 2. **Rescale (`update_a`, `fastmap.cpp:564-577`).** Only if `-A` was given. Each of
///    `b, t, o_del, o_ins, e_del, e_ins, zdrop, pen_clip5, pen_clip3, pen_unpaired` is multiplied by
///    the new `a`, *unless the user set that option too*. This is why every scalar arg is
///    `Option<T>`: `args.mismatch.is_none()` is precisely the C's `!opt0.b`.
/// 3. **Fill the matrix.** `opt.fill_scmat()` LAST, from the post-rescale `a`/`b`. Doing it in phase 1
///    would leave the SIMD kernel scoring with pre-rescale values while the gap logic used
///    post-rescale scalars: silently wrong alignments, but only on runs that pass `-A`.
///
/// # Parameters
///
/// - `args`: the whole parsed command line, borrowed read-only. Read twice: once in phase 1 for the
///   VALUES, and again in phase 2 for the `is_none()` PRESENCE tests that stand in for the C's
///   `opt0`. Both reads must see the same `args`, which is why it is borrowed rather than consumed
///   field by field.
///
/// # Returns
///
/// A `MemOpt` with every field final: defaults where the user was silent, user values where not,
/// rescaled by `-A` where phase 2 applied, and the scoring matrix filled from the final `a`/`b`.
/// Nothing downstream mutates it. The only errors come from the `INT[,INT]` sub-parsers.
///
/// UNVERIFIED: the C runs the `-x` preset block *between* phases 1 and 2 and, for a preset, takes a
/// different branch instead of `update_a` (`fastmap.cpp:820-860`). This port has no `-x`, so that
/// interaction is out of scope and has not been checked against the C in detail.
pub fn build_opt(args: &MemArgs) -> anyhow::Result<MemOpt> {
    // Start from bwa's compiled-in defaults (`mem_opt_init`), then overwrite only what the user
    // asked for. Every field not touched below keeps its default, and phase 2 relies on exactly
    // that: "still default" is what makes a field eligible for `-A` rescaling.
    let mut opt = MemOpt::default();
    // `-t`, clamped to >= 1 as both implementations do. Speed only, no effect on output bytes.
    opt.n_threads = args.threads.max(1);

    // ================= PHASE 1: apply every explicitly-given option =================
    //
    // Each `if let Some(v)` below reads "the user passed this flag": `None` is the C's `opt0` zero.
    // Every value here is verbatim what was typed, un-derived and un-rescaled; the only exception is
    // `mapq_coef_fac`, which the C itself derives inside the same getopt branch.

    // ---- Seeding and chaining ----
    // `-k`: shortest exact match that may start an alignment, in bases (default 19).
    if let Some(v) = args.min_seed_len {
        opt.min_seed_len = v;
    }
    // `-w`: band width in bases (default 100). How far off the diagonal the DP may wander, so it
    // caps the indel length a single extension can find. A length, never rescaled by `-A`.
    if let Some(v) = args.band_width {
        opt.w = v;
    }
    // `-d`: Z-drop in score units (default 100). Abandon an extension once its score falls this far
    // below its own best. In score units, so phase 2 may rescale it.
    if let Some(v) = args.zdrop {
        opt.zdrop = v;
    }
    // `-r`: re-seed trigger, dimensionless (default 1.5). A SMEM longer than `min_seed_len * this`
    // is re-seeded from its middle.
    if let Some(v) = args.split_factor {
        opt.split_factor = v;
    }
    // `-y`: occurrence threshold for the third seeding round (default 20). i64 because the C reads
    // it with `atol`.
    if let Some(v) = args.max_mem_intv {
        opt.max_mem_intv = v;
    }
    // `-c`: the repeat guard, in occurrences (default 500). Seeds occurring more often are skipped.
    if let Some(v) = args.max_occ {
        opt.max_occ = v;
    }
    // `-D`: chain drop ratio, a fraction in 0.0..=1.0 (default 0.50). Reused at output time as the
    // floor for emitting a secondary under `-a`, see `finish_se`.
    if let Some(v) = args.drop_ratio {
        opt.drop_ratio = v;
    }
    // `-W`: minimum seeded bases per chain (default 0, off).
    if let Some(v) = args.min_chain_weight {
        opt.min_chain_weight = v;
    }
    // `-m`: cap on mate-rescue rounds per read (default 50). Paired-end only, moot under `-S`.
    if let Some(v) = args.max_matesw {
        opt.max_matesw = v;
    }
    // `-G`: largest gap in bases between two seeds of one chain (default 10000), so it bounds the
    // deletion one record can span.
    if let Some(v) = args.max_chain_gap {
        opt.max_chain_gap = v;
    }
    // `-N`: cap on chain extensions (default 1<<30, effectively unlimited).
    if let Some(v) = args.max_chain_extend {
        opt.max_chain_extend = v;
    }
    // `-s`: re-seeding only fires when the long SMEM occurs fewer than this many times (default 10).
    if let Some(v) = args.split_width {
        opt.split_width = v;
    }
    // `-X`: redundancy overlap fraction, 0.0..=1.0 (default 0.50). A float, so never rescaled; the
    // C also never records it in `opt0`.
    if let Some(v) = args.mask_level {
        opt.mask_level = v;
    }
    // `-Q`: MAPQ coefficient length in bases (default 50).
    if let Some(v) = args.mapq_coef_len {
        // Two fields from one option: the C sets `mapQ_coef_len` and derives `mapQ_coef_fac` in the
        // same getopt branch (`fastmap.cpp:719-723`), so they can never drift apart.
        opt.mapq_coef_len = f64::from(v);
        // bwa: `mapQ_coef_fac = len > 0 ? log(len) : 0`, stored in an int (so truncated).
        opt.mapq_coef_fac = if v > 0 {
            f64::from(v).ln().trunc()
        } else {
            0.0
        };
    }

    // ---- Scoring scalars. `a` is the match reward, `b` the mismatch penalty (positive
    //      magnitude), `t` the minimum score worth emitting ----
    // `-A`: the match reward (default 1). Setting it is what arms phase 2 below.
    if let Some(v) = args.match_score {
        opt.a = v;
    }
    // `-B`: mismatch penalty as a positive magnitude (default 4); the `a`:`b` ratio sets tolerated
    // divergence.
    if let Some(v) = args.mismatch {
        opt.b = v;
    }
    // `-U`: penalty for reporting a pair as unpaired (default 17). Paired-end only, inert under -P.
    if let Some(v) = args.pen_unpaired {
        opt.pen_unpaired = v;
    }
    // `-T`: minimum score worth writing (default 30). Output filter only: it never changes which
    // alignments are found, and a read with nothing above it still gets an unmapped record.
    if let Some(v) = args.min_score {
        opt.t = v;
    }

    // ---- The four `INT[,INT]` options. `unwrap_or(first)` is the Rust spelling of the C's
    //      `opt->o_del = opt->o_ins = strtol(...)` then a conditional overwrite of the second ----
    // `o_*` is gap OPEN (charged once per gap), `e_*` gap EXTEND (charged per gap base); `_del` is a
    // gap in the read against the reference, `_ins` extra bases in the read.
    if let Some(s) = &args.gap_open {
        // `-O`: gap-open magnitudes in score units (default 6,6), deletion then insertion. A second
        // value is optional; absent, the first applies to both.
        let (open_del, open_ins) = parse_int_pair(s)?;
        opt.o_del = open_del;
        opt.o_ins = open_ins.unwrap_or(open_del);
    }
    if let Some(s) = &args.gap_extend {
        // `-E`: gap-extend magnitudes per gap base (default 1,1), so a length-k gap costs o + e*k.
        let (extend_del, extend_ins) = parse_int_pair(s)?;
        opt.e_del = extend_del;
        opt.e_ins = extend_ins.unwrap_or(extend_del);
    }
    if let Some(s) = &args.clip_penalty {
        // `-L`: clipping penalties (default 5,5), 5' end then 3'. Steers local vs glocal only; not
        // deducted from the reported AS:i score.
        let (clip5, clip3) = parse_int_pair(s)?;
        opt.pen_clip5 = clip5;
        opt.pen_clip3 = clip3.unwrap_or(clip5);
    }
    if let Some(s) = &args.xa_hits {
        // `-h`: XA:Z listing limits in hit counts (default 5,200), primary assembly then ALT. Counts,
        // not scores, so phase 2 never touches them. The ALT limit is currently unreachable, see -j.
        let (xa_hits, xa_hits_alt) = parse_int_pair(s)?;
        opt.max_xa_hits = xa_hits;
        opt.max_xa_hits_alt = xa_hits_alt.unwrap_or(xa_hits);
    }

    // ---- Behaviour flags (the MEM_F_* bits of `opt.flag`). Each is a boolean the user either
    //      passed or did not: there is no "unset" state to preserve, so none of them interacts with
    //      phase 2. `flag` starts at the default (0 for every bit) and only ever gains bits here ----
    // `-P`: score each mate on its own, never jointly.
    if args.skip_pairing {
        opt.flag |= flags::NOPAIRING;
    }
    // `-a`: emit shadowed hits as their own secondary records instead of folding them into XA:Z.
    if args.output_all {
        opt.flag |= flags::ALL;
    }
    // `-p`: one interleaved input file. Sets PE too, so this alone switches on paired-end mode.
    if args.smart_pairing {
        opt.flag |= flags::PE | flags::SMARTPE;
    }
    // `-M`: report supplementary records as secondary, for tools predating FLAG 0x800.
    if args.mark_secondary {
        opt.flag |= flags::NO_MULTI;
    }
    // `-S`: skip mate rescue (roughly half of paired-end wall time here).
    if args.skip_mate_rescue {
        opt.flag |= flags::NO_RESCUE;
    }
    // `-Y`: soft-clip supplementary records instead of hard-clipping them.
    if args.soft_clip_supp {
        opt.flag |= flags::SOFTCLIP;
    }
    // `-V`: emit the reference FASTA header in the XR tag.
    if args.ref_hdr {
        opt.flag |= flags::REF_HDR;
    }
    // bwa always applies KEEP_SUPP_MAPQ together with -5.
    if args.primary5 {
        opt.flag |= flags::PRIMARY5 | flags::KEEP_SUPP_MAPQ;
    }
    // `-q`: leave a supplementary record's MAPQ uncapped. Already implied by `-5` above; setting it
    // twice is idempotent, which is why the C can keep the two branches separate.
    if args.keep_supp_mapq {
        opt.flag |= flags::KEEP_SUPP_MAPQ;
    }

    // ================= PHASE 2: `update_a` rescaling =================
    //
    // A changed match score rescales every penalty left at its default, so the ratios that give bwa
    // its tuned behaviour survive.
    //
    // The guard is `opt0->a`, i.e. "was -A given", NOT "is a != 1". Passing `-A 1` explicitly still
    // enters this block; it just multiplies by 1, so the result is identical. Order inside the block
    // is irrelevant (each field is independent), but the block as a whole must run after every
    // option has been applied and before the matrix is filled.
    // The C's shape is `if (mode) { presets } else update_a(opt, &opt0);` (`fastmap.cpp:818-860`),
    // and the `else` is load-bearing: **a preset SUPPRESSES the `-A` rescaling entirely**, even when
    // `-A` was given. Writing this as two independent `if`s would rescale a preset's values and
    // diverge from bwa on `-x pacbio -A 2`.
    //
    // Reachability: of the four modes below, only `intractg` is reached through `run`, which routes
    // `pacbio`/`pbref`/`ont2d` to rammap before this function is called (see `cmd_longread`). The
    // other three branches are kept, and kept correct, for two reasons: `build_opt` is a pure
    // function whose contract is "bwa's option semantics", and the routing is a policy decision that
    // a future release could revisit. `presets_match_bwa_long_read` exercises them directly, so they
    // cannot rot silently while unreachable from the binary's main path.
    if let Some(mode) = args.read_type.as_deref() {
        // Same text and the same stream as bwa-mem2 (`fastmap.cpp:820`). stderr only, so it cannot
        // reach the SAM bytes.
        eprintln!(
            "WARNING: bwa-mem2 doesn't work well with long reads or contigs; please use minimap2 \
             instead."
        );
        match mode {
            "intractg" => {
                // Intra-species contig alignment: expensive gaps, harsh mismatches, mild clipping.
                if args.gap_open.is_none() {
                    opt.o_del = 16;
                    opt.o_ins = 16;
                }
                if args.mismatch.is_none() {
                    opt.b = 9;
                }
                if args.clip_penalty.is_none() {
                    opt.pen_clip5 = 5;
                    opt.pen_clip3 = 5;
                }
            }
            // `pbref` is bwa's third spelling of the pacbio preset; the C tests all three in one
            // branch and only splits `ont2d` out afterwards.
            "pacbio" | "pbref" | "ont2d" => {
                // Long, error-prone reads: gaps and mismatches nearly free, no clipping penalty,
                // and a much more aggressive re-seed.
                if args.gap_open.is_none() {
                    opt.o_del = 1;
                    opt.o_ins = 1;
                }
                if args.gap_extend.is_none() {
                    opt.e_del = 1;
                    opt.e_ins = 1;
                }
                if args.mismatch.is_none() {
                    opt.b = 1;
                }
                // The C's guard is `opt0.split_factor == 0.`, a float compare against the flag it
                // sets to `1.` in the `-r` branch; `None` here is the same predicate.
                if args.split_factor.is_none() {
                    opt.split_factor = 10.0;
                }
                let ont = mode == "ont2d";
                if args.min_chain_weight.is_none() {
                    opt.min_chain_weight = if ont { 20 } else { 40 };
                }
                if args.min_seed_len.is_none() {
                    opt.min_seed_len = if ont { 14 } else { 17 };
                }
                if args.clip_penalty.is_none() {
                    opt.pen_clip5 = 0;
                    opt.pen_clip3 = 0;
                }
            }
            other => {
                // The C prints `[E::main_mem] unknown read type '<x>'` and returns 1.
                anyhow::bail!("unknown read type '{other}'");
            }
        }
    } else if args.match_score.is_some() {
        if args.mismatch.is_none() {
            opt.b *= opt.a;
        }
        if args.min_score.is_none() {
            opt.t *= opt.a;
        }
        if args.gap_open.is_none() {
            opt.o_del *= opt.a;
            opt.o_ins *= opt.a;
        }
        if args.gap_extend.is_none() {
            opt.e_del *= opt.a;
            opt.e_ins *= opt.a;
        }
        if args.zdrop.is_none() {
            opt.zdrop *= opt.a;
        }
        if args.clip_penalty.is_none() {
            opt.pen_clip5 *= opt.a;
            opt.pen_clip3 *= opt.a;
        }
        if args.pen_unpaired.is_none() {
            opt.pen_unpaired *= opt.a;
        }
    }

    // ================= PHASE 3: fill the scoring matrix, LAST =================
    //
    // From the POST-rescale a/b. Doing this earlier would leave the SIMD kernel scoring with
    // pre-rescale values while the gap logic used post-rescale scalars.
    opt.fill_scmat();
    Ok(opt)
}

/// The SAM output sink: plain (stdout or an uncompressed file), parallel BGZF (block-gzip), or a
/// binary BAM/CRAM transcoder. All expose `std::io::Write` so the formatting/writing paths stay
/// generic; `finish` drains the BGZF worker pool and writes the EOF marker (surfacing errors the
/// `Drop` path would swallow), and closes the htslib file.
///
/// The binary variants deliberately do NOT have their own record formatter. Every record is still
/// formatted as SAM text by exactly the code that feeds the `Plain` sink, and [`HtsTranscoder`]
/// re-encodes that text. Byte-identity with bwa-mem2 is defined on the SAM bytes, so a second,
/// independent formatting path into BAM would be a second thing to keep byte-identical, and one
/// that no parity test covers. Transcoding makes BAM a lossless re-encoding of the bytes we already
/// prove correct.
enum Output {
    /// Uncompressed SAM: a `BufWriter` over stdout (no `-o`) or over the named file. This is the
    /// only variant bwa itself has. `finish` just flushes it.
    Plain(Box<dyn Write + Send>),
    /// BGZF, selected by a `.gz`/`.bgz` suffix on `-o`. Wraps a buffered file sink and compresses on
    /// `-t` worker threads. `finish` must run: it drains those workers and writes the 28-byte EOF
    /// block that samtools requires of a complete BGZF file.
    Bgzf(Box<bgzf::MultithreadedWriter<Box<dyn Write + Send>>>),
    /// Binary BAM, selected by a `.bam` suffix on `-o`. Boxed because a `bam::Writer` plus the
    /// transcoder's line buffers dwarf the other variants' two pointers.
    #[cfg(feature = "hts-output")]
    Bam(Box<HtsTranscoder>),
    /// CRAM, selected by a `.cram` suffix on `-o`. Identical machinery to `Bam` (only the htslib
    /// format flag and the mandatory `--reference` differ), which is why both hold the same type.
    #[cfg(feature = "hts-output")]
    Cram(Box<HtsTranscoder>),
}

/// Streaming SAM-text -> BAM/CRAM transcoder.
///
/// It sits behind `impl Write for Output` because that is the shape the pipeline hands us: the
/// writer thread ships whole batch buffers of SAM text (`run_pipeline`), each holding many records,
/// and the header arrives as a separate direct write before the pipeline even starts. So this is a
/// byte-stream state machine, not a record API:
///
/// 1. Bytes accumulate in `pending` until a `\n` completes a line. A batch buffer always ends on a
///    record boundary today, but nothing in the pipeline's contract guarantees that, so a partial
///    trailing line is carried across `write` calls rather than assumed away.
/// 2. While no writer is open, lines starting with `@` are appended to `header_text`. A record line
///    can never be mistaken for a header line: SAM restricts QNAME's first character to
///    `[!-?A-~]`, which excludes `@` precisely so that this test is unambiguous.
/// 3. The first non-`@` line closes the header: htslib parses `header_text`, the file is opened,
///    and from then on every line goes through `bam::Record::from_sam`.
///
/// Opening lazily (step 3) is what lets the header be written verbatim, as one blob of the exact
/// bytes `sam::write_header` produced, instead of being rebuilt tag by tag from `sqs` and risking a
/// divergence from the SAM path.
#[cfg(feature = "hts-output")]
struct HtsTranscoder {
    /// Where to open, kept because the open is deferred to the first record line.
    path: PathBuf,
    /// htslib's `hts_open` mode string: `wb` for BAM, `wc` for CRAM (`htslib/hts.h`). The only real
    /// difference between the `Bam` and `Cram` variants.
    mode: &'static std::ffi::CStr,
    /// `--reference` FASTA, required for CRAM (which stores bases as differences against it) and
    /// unused by BAM. Presence is checked in [`Output::open`], applied at file-open time.
    reference: Option<PathBuf>,
    /// htslib background compression threads, from `-t`.
    threads: usize,
    /// The `@` lines seen so far, newline-terminated, exactly as the SAM path emitted them. Becomes
    /// the htslib header when the first record line arrives.
    header_text: Vec<u8>,
    /// Bytes after the last `\n` of the previous `write` call: the incomplete tail of a line.
    pending: Vec<u8>,
    /// The open file, `None` until the header is complete.
    writer: Option<HtsWriter>,
}

#[cfg(feature = "hts-output")]
impl HtsTranscoder {
    /// Feed one complete line, without its trailing `\n`. Header lines are buffered; the first
    /// record line opens the file, and every record line is parsed and written.
    fn consume_line(&mut self, line: &[u8]) -> anyhow::Result<()> {
        // Defensive: our formatter never emits a blank line, but a blank one would make `from_sam`
        // fail with an opaque parse error rather than being harmlessly skipped.
        if line.is_empty() {
            return Ok(());
        }
        if self.writer.is_none() && line[0] == b'@' {
            self.header_text.extend_from_slice(line);
            self.header_text.push(b'\n');
            return Ok(());
        }
        let writer = self.open_writer()?;
        // `from_sam` is htslib's own `sam_parse1`, i.e. the same parser samtools uses when it reads
        // our SAM file back. That equivalence is the point: whatever samtools would make of the
        // text, the BAM/CRAM holds.
        let rec = bam::Record::from_sam(&writer.header, line)?;
        writer.write(&rec)?;
        Ok(())
    }

    /// The writer, opening the file on first use now that `header_text` is complete.
    fn open_writer(&mut self) -> anyhow::Result<&mut HtsWriter> {
        if self.writer.is_none() {
            self.writer = Some(HtsWriter::open(
                &self.path,
                self.mode,
                self.reference.as_deref(),
                self.threads,
                &self.header_text,
            )?);
        }
        // The `expect` is unreachable: the branch above either filled `writer` or returned an error.
        Ok(self.writer.as_mut().expect("writer just opened"))
    }

    /// Flush and close. Consumes the transcoder.
    fn finish(mut self) -> anyhow::Result<()> {
        // A trailing byte run with no `\n` cannot be completed by anything now, so it is a whole
        // final line by definition. Our formatter always newline-terminates, so this is normally
        // empty; feeding it is still better than silently dropping a record.
        let tail = std::mem::take(&mut self.pending);
        if !tail.is_empty() {
            self.consume_line(&tail)?;
        }
        // A run that produced no records at all still owes the caller a valid, header-only file.
        self.open_writer()?;
        // `close` is where htslib finalizes: BAM's 28-byte EOF block, CRAM's end-of-file container.
        // Taken out of the option so the `Drop` below finds nothing left to close.
        self.writer.take().expect("writer just opened").close()
    }
}

/// An open htslib output file: `hts_open` + `sam_hdr_write` + `sam_write1` + `hts_close`.
///
/// This is deliberately NOT `rust_htslib::bam::Writer`, and the reason is CRAM. `Writer::new` writes
/// the header inside the constructor, so the earliest a caller can call `Writer::set_reference` is
/// AFTER `sam_hdr_write`, and that is too late: `cram_write_SAM_hdr`
/// (`htslib/cram/cram_io.c`, the "Fix M5 strings" block) walks the @SQ lines, finds neither an `M5`
/// tag nor a loaded reference, and permanently flips the file descriptor to `embed_ref=2`
/// (auto-generate an embedded reference). Every later container then calls
/// `cram_generate_reference`, which fails on our read-ordered (not coordinate-sorted) output with
/// "Cannot build reference with unsorted data" and falls back to non-reference CRAM. That still
/// round-trips correctly, but it throws away the entire point of CRAM: measured on 50k pairs, 5.0 MB
/// non-ref versus 1.0 MB with the external reference. Adding `M5` to our @SQ lines would also
/// prevent the flip, but the SAM text is the parity oracle and may not gain a tag.
///
/// So the sequence below is `hts_open` -> `hts_set_fai_filename` -> `sam_hdr_write`, which is exactly
/// what `samtools view -T ref.fa -C` does (`sam_view.c` sets the reference before writing the
/// header). Everything else still comes from rust-htslib: the header parse (`HeaderView`) and the
/// SAM-to-record parse (`Record::from_sam`) are its types, and we only borrow their raw pointers.
#[cfg(feature = "hts-output")]
struct HtsWriter {
    /// The htslib file. Non-null for the whole life of the value; nulled by `close` so that `Drop`
    /// does not double-close.
    fp: *mut htslib::htsFile,
    /// The parsed header. Owned here because `sam_write1` needs it on every record (it maps RNAME
    /// to a numeric tid) and `Record::from_sam` needs it to parse one.
    header: bam::HeaderView,
}

// SAFETY: `htsFile` is not shared, only moved: the sink is created on the main thread and handed to
// the writer thread, which is then the only thread that touches it. This is the same reasoning (and
// the same guarantee) as rust-htslib's own `unsafe impl Send for Writer`.
#[cfg(feature = "hts-output")]
unsafe impl Send for HtsWriter {}

#[cfg(feature = "hts-output")]
impl HtsWriter {
    /// Open `path` for writing in `mode`, install `reference` if given, then write `header_text`
    /// verbatim as the file header. The order of those three steps is load-bearing; see the type
    /// docs.
    fn open(
        path: &Path,
        mode: &'static std::ffi::CStr,
        reference: Option<&Path>,
        threads: usize,
        header_text: &[u8],
    ) -> anyhow::Result<Self> {
        // Paths cross the FFI boundary as NUL-terminated bytes; an embedded NUL is the one thing a
        // Rust path can hold that a C string cannot.
        let cpath = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| anyhow::anyhow!("-o {}: path contains a NUL byte", path.display()))?;
        // SAFETY: both pointers are valid NUL-terminated C strings that outlive the call, and
        // htslib copies what it needs from them.
        let fp = unsafe { htslib::hts_open(cpath.as_ptr(), mode.as_ptr()) };
        if fp.is_null() {
            anyhow::bail!("-o {}: htslib could not open the file", path.display());
        }
        // From here on every early return must not leak `fp`, so the guard is built immediately and
        // its `Drop` owns the close. `header` is a placeholder only until `sam_hdr_write` below;
        // parsing it here (rather than after) is what lets that be true.
        let this = HtsWriter {
            fp,
            header: bam::HeaderView::from_bytes(header_text),
        };
        if let Some(reference) = reference {
            let cref = std::ffi::CString::new(reference.as_os_str().as_encoded_bytes())
                .map_err(|_| anyhow::anyhow!("--reference: path contains a NUL byte"))?;
            // SAFETY: `this.fp` is a live htsFile and `cref` outlives the call. htslib accepts
            // either the FASTA or its `.fai` here (`refs_load_fai` strips the suffix); we pass the
            // FASTA, as `samtools -T` does, so that a missing `.fai` is merely slow (htslib scans
            // the FASTA to build the index in memory) rather than fatal.
            if unsafe { htslib::hts_set_fai_filename(this.fp, cref.as_ptr()) } != 0 {
                anyhow::bail!(
                    "--reference {}: htslib could not load it as a CRAM reference",
                    reference.display()
                );
            }
        }
        // htslib's own background compressor, sized like the BGZF variant's worker pool. Purely a
        // throughput knob: it changes block boundaries, never the decoded records.
        // SAFETY: live htsFile, and a positive thread count is what htslib documents.
        if unsafe { htslib::hts_set_threads(this.fp, threads.max(1) as std::os::raw::c_int) } != 0 {
            anyhow::bail!("htslib rejected {} compression threads", threads.max(1));
        }
        // SAFETY: live htsFile and a header parsed by htslib itself. `sam_hdr_write` only reads the
        // header, hence the const pointer.
        if unsafe { htslib::sam_hdr_write(this.fp, this.header.inner_ptr()) } < 0 {
            anyhow::bail!("-o {}: writing the header failed", path.display());
        }
        Ok(this)
    }

    /// Write one record. The header must be the one it was parsed against: `sam_write1` reads the
    /// record's numeric `tid`, which only means anything relative to that header's @SQ order.
    fn write(&mut self, rec: &bam::Record) -> anyhow::Result<()> {
        // SAFETY: all three pointers are live, and htslib only reads the header and the record.
        if unsafe { htslib::sam_write1(self.fp, self.header.inner_ptr(), rec.inner()) } < 0 {
            anyhow::bail!("writing a BAM/CRAM record failed");
        }
        Ok(())
    }

    /// Finalize and close, surfacing the error `Drop` would have to swallow. `hts_close` is where
    /// the last block is compressed and flushed, so this is the call that reports a full disk.
    fn close(mut self) -> anyhow::Result<()> {
        // Nulled first so the `Drop` that runs at the end of this function sees nothing to do.
        let fp = std::mem::replace(&mut self.fp, std::ptr::null_mut());
        // SAFETY: `fp` is live and is not used again (the field is now null).
        if unsafe { htslib::hts_close(fp) } != 0 {
            anyhow::bail!("closing the BAM/CRAM file failed (output may be truncated)");
        }
        Ok(())
    }
}

#[cfg(feature = "hts-output")]
impl Drop for HtsWriter {
    fn drop(&mut self) {
        // Only reached on an error path, since `finish` closes explicitly. The file still has to be
        // closed, but there is no one left to report to, so the return value is discarded here (and
        // ONLY here).
        if !self.fp.is_null() {
            // SAFETY: non-null means still open, and nothing touches `fp` after this.
            unsafe { htslib::hts_close(self.fp) };
        }
    }
}

/// BGZF deflate level. The `bgzf` crate accepts 0-12; 6 is zlib's traditional default and is what
/// `bgzip` and `samtools` write, so our `.gz` output is comparable in size to theirs. Not a
/// correctness parameter: any level decompresses to the same bytes.
const BGZF_COMPRESSION_LEVEL: u8 = 6;

/// 1 MiB of buffering under the BGZF writer, so the compressor's block-sized writes do not each
/// become a syscall. Sized here, not inherited from bwa (which never writes compressed SAM).
const BGZF_SINK_BUFFER_BYTES: usize = 1 << 20;

/// 1 MiB of buffering under the uncompressed sink, for the same reason and for one more.
///
/// `BufWriter`'s 8 KiB default was harmless while a batch arrived as ONE `Vec<u8>`: a write larger
/// than the buffer bypasses it entirely, so a 400 MB batch was a single `write`. Now that a batch
/// arrives as per-record pieces of a few hundred bytes each (see `run_pipeline`), the buffer is
/// what turns them back into few, large syscalls. At 8 KiB a `-t16` batch would be ~50k of them;
/// at 1 MiB it is ~400. Output bytes are identical either way, only the syscall count changes.
const PLAIN_SINK_BUFFER_BYTES: usize = 1 << 20;

impl Write for Output {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Output::Plain(w) => w.write(buf),
            Output::Bgzf(w) => w.write(buf),
            // Both binary variants consume the whole buffer or fail: a partial count would leave the
            // transcoder's line state and the caller's cursor disagreeing about what was consumed.
            #[cfg(feature = "hts-output")]
            Output::Bam(t) | Output::Cram(t) => {
                // Split on `\n`, carrying an unterminated tail into `pending` for the next call.
                // `rest` shrinks to the not-yet-dispatched remainder of `buf`.
                let mut rest = buf;
                while let Some(nl) = rest.iter().position(|&b| b == b'\n') {
                    let (line, after) = rest.split_at(nl);
                    let res = if t.pending.is_empty() {
                        // Whole line inside this buffer: pass the slice straight through, no copy.
                        t.consume_line(line)
                    } else {
                        // The line began in an earlier `write`. Taking `pending` out first both
                        // ends the borrow of `self` and leaves the field empty for the next line.
                        let mut joined = std::mem::take(&mut t.pending);
                        joined.extend_from_slice(line);
                        let res = t.consume_line(&joined);
                        // Reuse the allocation, now emptied, as the next partial-line buffer.
                        joined.clear();
                        t.pending = joined;
                        res
                    };
                    res.map_err(io::Error::other)?;
                    rest = &after[1..]; // skip the `\n` itself
                }
                t.pending.extend_from_slice(rest);
                Ok(buf.len())
            }
        }
    }
    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        match self {
            Output::Plain(w) => w.flush(),
            Output::Bgzf(w) => w.flush(),
            // htslib buffers internally and offers no flush short of closing the file, which
            // `finish` does. Nothing here can be forced out, so this is honestly a no-op.
            #[cfg(feature = "hts-output")]
            Output::Bam(_) | Output::Cram(_) => Ok(()),
        }
    }
}

impl Output {
    /// Open the output sink. `None` writes uncompressed SAM to stdout; a path with a `.gz`/`.bgz`
    /// suffix writes BGZF compressed in parallel on `threads` workers; `.bam`/`.cram` write binary
    /// BAM/CRAM through htslib; any other path writes an uncompressed file.
    ///
    /// # Parameters
    ///
    /// - `path`: the `-o` argument, or `None` for stdout. An existing file is truncated.
    /// - `threads`: BGZF compression workers, from `-t`. Clamped to >= 1, and ignored entirely for
    ///   the plain variants. It affects only throughput: BGZF block boundaries and therefore the
    ///   decompressed bytes are the same at any worker count. The same value is handed to htslib's
    ///   background compressor for `.bam`/`.cram`.
    /// - `reference`: the `--reference` FASTA. Required for `.cram` and ignored otherwise.
    ///
    /// # Returns
    ///
    /// The opened sink. Errors if the file cannot be created, if `.cram` was asked for without a
    /// `--reference`, or if the compile-time compression level is somehow out of the `bgzf` crate's
    /// accepted range. Note the binary variants do not touch the filesystem here: htslib opens the
    /// file only once the header text has streamed through (see [`HtsTranscoder`]).
    fn open(
        path: Option<&Path>,
        threads: usize,
        // Only CRAM reads it, so without `hts-output` there is nothing to read it.
        #[cfg_attr(not(feature = "hts-output"), allow(unused_variables))] reference: Option<&Path>,
    ) -> anyhow::Result<Self> {
        match path {
            None => Ok(Output::Plain(Box::new(BufWriter::with_capacity(
                PLAIN_SINK_BUFFER_BYTES,
                std::io::stdout(),
            )))),
            Some(path) => {
                // Binary formats first, by the same case-insensitive suffix rule as `.gz` below, and
                // before the `File::create` further down: htslib must be the one to open the file.
                let ext_is = |want: &str| {
                    path.extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case(want))
                };
                if ext_is("bam") || ext_is("cram") {
                    // Refused rather than quietly written as SAM text under a binary name, which
                    // would hand back a file every downstream tool rejects.
                    #[cfg(not(feature = "hts-output"))]
                    anyhow::bail!(
                        "-o {}: this build has no BAM/CRAM writer. Rebuild with \
                         `--features hts-output`, or write SAM (`-o out.sam`) or BGZF \
                         (`-o out.sam.gz`), which every samtools reads.",
                        path.display()
                    );
                    #[cfg(feature = "hts-output")]
                    let want_cram = ext_is("cram");
                    // Refused up front rather than at the first record: CRAM stores SEQ as a diff
                    // against the reference, so without one there is nothing to write.
                    #[cfg(feature = "hts-output")]
                    if want_cram && reference.is_none() {
                        anyhow::bail!(
                            "-o {}: CRAM output needs the reference FASTA, pass --reference <FASTA> \
                             (the same one you indexed; its .fai must exist)",
                            path.display()
                        );
                    }
                    #[cfg(feature = "hts-output")]
                    let transcoder = Box::new(HtsTranscoder {
                        path: path.to_path_buf(),
                        // htslib's write modes: `w` plus `b` for BAM or `c` for CRAM.
                        mode: if want_cram { c"wc" } else { c"wb" },
                        // Kept for BAM too, harmlessly: htslib ignores a reference for BAM, and
                        // dropping it here would make the two variants differ for no reason.
                        reference: reference.map(Path::to_path_buf),
                        threads,
                        header_text: Vec::new(),
                        pending: Vec::new(),
                        writer: None,
                    });
                    #[cfg(feature = "hts-output")]
                    return Ok(if want_cram {
                        Output::Cram(transcoder)
                    } else {
                        Output::Bam(transcoder)
                    });
                }
                // Whether the caller asked for BGZF, decided purely from the suffix (case-insensitive
                // `.gz`/`.bgz`), as bgzip and samtools do. There is no explicit flag for it.
                let want_bgzf = path.extension().is_some_and(|ext| {
                    ext.eq_ignore_ascii_case("gz") || ext.eq_ignore_ascii_case("bgz")
                });
                // Created before the branch so both paths share the truncate-or-fail behaviour.
                let file = std::fs::File::create(path)?;
                if want_bgzf {
                    // The buffered byte sink the compressor's block writes land in, type-erased so
                    // the `MultithreadedWriter` generic matches the `Plain` boxed writer.
                    let sink: Box<dyn Write + Send> =
                        Box::new(BufWriter::with_capacity(BGZF_SINK_BUFFER_BYTES, file));
                    // Validated wrapper around BGZF_COMPRESSION_LEVEL; the fallible constructor is
                    // why this is not a plain integer.
                    let level = bgzf::CompressionLevel::new(BGZF_COMPRESSION_LEVEL)
                        .map_err(|e| anyhow::anyhow!("bgzf compression level: {e}"))?;
                    // `threads` as the crate's non-zero type. The `unwrap_or` is unreachable given
                    // the `max(1)`, and just avoids a panic path.
                    let worker_count = std::num::NonZero::new(threads.max(1))
                        .unwrap_or(std::num::NonZero::<usize>::MIN);
                    Ok(Output::Bgzf(Box::new(
                        bgzf::MultithreadedWriter::with_worker_count(worker_count, sink, level),
                    )))
                } else {
                    Ok(Output::Plain(Box::new(BufWriter::with_capacity(
                        PLAIN_SINK_BUFFER_BYTES,
                        file,
                    ))))
                }
            }
        }
    }

    /// Flush and finalize (BGZF: drain workers + write the EOF marker; BAM/CRAM: write any
    /// unterminated final line, then close the htslib file, which is what emits BAM's EOF block or
    /// CRAM's end-of-file container). Consumes the sink.
    fn finish(self) -> anyhow::Result<()> {
        match self {
            Output::Plain(mut w) => w.flush()?,
            Output::Bgzf(mut w) => {
                w.finish()?;
            }
            #[cfg(feature = "hts-output")]
            Output::Bam(t) | Output::Cram(t) => t.finish()?,
        }
        Ok(())
    }
}

/// How many batches may be in flight, as `(readahead, writebehind)`.
///
/// One each way, which is double buffering: while the aligner works on a batch, the reader may have
/// exactly one more ready and the writer may have exactly one still draining. Nothing deeper.
///
/// It used to be 2 and 3, and those were the whole of the memory gap against `fg-labs/bwa-mem3`
/// reported in issue #25. The queues are bounded in BATCHES, and a batch is `-K` bases, which at the
/// default `-K` of 10M PER THREAD grows with `-t`. So the two constants meant something different at
/// every invocation: measured on a 200 kb reference so the index contributes nothing to the peak,
/// they cost about 100 MB at `-K` 10M and about 430 MB at `-K` 100M, all of it records and SAM text
/// that nothing was reading yet. At a small `-K` we were already LIGHTER than the fork, which is
/// what says the excess tracked the queue depth and not the per-read state the issue's Mechanism
/// section named.
///
/// The depth those constants bought was not paying for itself. On chr21 with 500k pairs at `-t8`,
/// native arm64, going from (2,3) to (1,1): plain FASTQ 19.72s -> 20.02s, gzipped input 15.32s ->
/// 14.90s. Both differences are inside the run-to-run spread, and the gzipped case, the one a deeper
/// queue is supposed to protect, came out slightly faster. A queue that does not hide latency is
/// just memory.
///
/// `BWA4_QUEUE_DEPTH=read,write` raises them again, so the trade can be measured on a machine where
/// input really is the bottleneck rather than argued about here. A value that does not parse, or
/// that asks for a zero-length queue, is ignored rather than clamped silently: a typo in a benchmark
/// script should not quietly change what is being measured.
const BATCH_READAHEAD: usize = 1;
const SAM_WRITEBEHIND: usize = 1;

fn queue_depths() -> (usize, usize) {
    if let Ok(s) = std::env::var("BWA4_QUEUE_DEPTH") {
        if let Some((r, w)) = s.split_once(',') {
            if let (Ok(r), Ok(w)) = (r.trim().parse::<usize>(), w.trim().parse::<usize>()) {
                if r >= 1 && w >= 1 {
                    return (r, w);
                }
            }
        }
    }
    (BATCH_READAHEAD, SAM_WRITEBEHIND)
}

/// The single-end reader's body, as a closure the caller hands to [`spawn_reader`] before loading
/// the index. The paired twin is [`pe_read_batches`]; both exist for the same reason, which is that
/// the first batch's read, inflate and parse should not queue behind a 10 GB index load.
///
/// # Parameters
/// * `reads_path`: the FASTQ, gzipped or not. Owned, so the closure borrows nothing.
/// * `k_batch`: `-K` in input bases; fixes the batch boundaries and therefore the output.
fn se_read_batches(
    reads_path: std::path::PathBuf,
    k_batch: usize,
) -> impl FnOnce(std::sync::mpsc::SyncSender<(Vec<Record>, u64)>) -> anyhow::Result<()> + Send + 'static
{
    move |tx: std::sync::mpsc::SyncSender<(Vec<Record>, u64)>| -> anyhow::Result<()> {
        // Opened INSIDE the closure so the reader itself never crosses a thread boundary.
        let mut reader = FastqReader::from_path(&reads_path)?;
        // Number of reads emitted in all previous batches, i.e. the global 0-based id of this
        // batch's first read. Invariant at the top of each turn: it equals the total length of every
        // batch already sent. Must stay global across batches, since downstream tie-breaks hash it.
        let mut base_id = 0u64;
        loop {
            // Up to `k_batch` INPUT BASES worth of reads; empty only at end of file.
            let batch = reader.next_batch(k_batch)?;
            if batch.is_empty() {
                break;
            }
            // Read count, saved before the move so `base_id` can advance after the send.
            let n = batch.len() as u64;
            if tx.send((batch, base_id)).is_err() {
                break;
            }
            base_id += n;
        }
        Ok(())
    }
}

/// Overlap I/O with compute. A **reader** thread produces batches (opening the FASTQ *inside* the
/// thread, so the reader never crosses a thread boundary: only the `Send` record batches do), the
/// main thread aligns+formats each batch (internally parallel across the rayon pool) into one byte
/// buffer, and a **writer** thread drains those buffers. Bounded channels cap the batches in flight.
///
/// Output order equals read order (one reader, then the sequential main thread, then one writer),
/// and batch boundaries are fixed by `-K`, so the output is byte-identical to the old serial
/// read/align/write loop. The serial read and write are simply hidden behind the next batch's
/// compute.
///
/// # Parameters
///
/// - `B`: the batch payload, `Vec<Record>` single-end or `Vec<(Record, Record)>` paired-end.
/// - `out`: the already-opened sink, MOVED into the writer thread (it is only ever touched there,
///   which is what lets `Output` be non-`Sync`). `finish` runs on the writer, after the last batch.
/// - `read_batches`: runs on the reader thread. Sends `(batch, base_id)` where `base_id` is the
///   cumulative count of reads (SE) or pairs (PE) emitted before this batch. That id must be global
///   across batches: downstream tie-breaks hash it, so restarting it per batch would change output.
/// - `process`: runs on the main thread and returns one batch's SAM bytes AS PER-RECORD PIECES, in
///   output order. Internally parallel over the rayon pool; must be a pure function of its
///   arguments for the byte-identity claim to hold.
///
///   The pieces are handed over as a `Vec<Vec<u8>>` rather than concatenated into one buffer,
///   because concatenating cost a single-threaded memcpy of the whole batch's SAM text (~400 MB at
///   `-t16`) plus the first-touch page faults on a fresh allocation that large, and it made the
///   text momentarily resident TWICE at exactly the instant that sets peak RSS. The writer loops
///   over the pieces instead. This cannot change the output: the byte stream is the same pieces in
///   the same order, and every `Output` variant is chunk-boundary agnostic (`Plain` and `Bgzf`
///   buffer, and the `Bam`/`Cram` transcoder already carries an unterminated line across `write`
///   calls in `pending`). It is the same shape as bwa-mem2, which `fputs`es each `seqs[i].sam`
///   individually rather than building a batch buffer (`fastmap.cpp`).
///
/// # Returns
///
/// `Ok(())` only once BOTH helper threads have joined cleanly and the sink has been finalized. A
/// reader or writer error is reported at `join`, so it is never lost even though the send sites
/// merely `break`.
///
/// # Invariant
///
/// Exactly one reader and one writer, and the main thread consumes batches sequentially, so record
/// order in the output file equals record order in the input. Errors on either thread surface at
/// `join`, which is why the send failures below only `break` rather than reporting.
/// Start the reader thread NOW, before the caller loads the index.
///
/// The reader opens the FASTQ, inflates it if it is gzipped, and parses records; on a whole-genome
/// run at the default `-K` the first batch is 160M bases and that work is on the order of a second.
/// The index load is another 1.8 s. Done in sequence, as they were, the run pays for both before the
/// first alignment starts; started here, they overlap and the run pays for the longer of the two.
///
/// The reader owns everything it touches (an owned path, the batch size) so it needs no borrow from
/// the caller and can outlive this call. The channel is bounded at the same readahead depth the
/// pipeline uses, so the reader still cannot run ahead and accumulate batches while the index is
/// loading: it fills the queue and blocks, which is exactly the intended behaviour.
///
/// Output is unaffected. The reader produces the same batches in the same order with the same base
/// ids; only the moment it starts changes.
///
/// # Returns
/// The receiving half to hand to [`run_pipeline`], and the join handle, which must be joined after
/// the pipeline so a reader error is not silently dropped.
fn spawn_reader<B: Send + 'static>(
    read_batches: impl FnOnce(std::sync::mpsc::SyncSender<(B, u64)>) -> anyhow::Result<()>
        + Send
        + 'static,
) -> (
    std::sync::mpsc::Receiver<(B, u64)>,
    std::thread::JoinHandle<anyhow::Result<()>>,
) {
    let (readahead, _) = queue_depths();
    let (tx, rx) = std::sync::mpsc::sync_channel::<(B, u64)>(readahead);
    (
        rx,
        std::thread::spawn(move || {
            // Tagged so `BWA4_STAGE_ALLOC` credits decompression and record parsing to `reader`
            // rather than to whichever stage the aligner happened to be in when they ran. No-op
            // unless that probe is armed.
            crate::stage_alloc::set_role(crate::stage_alloc::ROLE_READER);
            read_batches(tx)
        }),
    )
}

fn run_pipeline<B: Send>(
    out: Output,
    // Already produced by a reader the CALLER started, so that reading and inflating the first
    // batch overlaps the index load instead of queueing behind it. See `spawn_reader`.
    batch_rx: std::sync::mpsc::Receiver<(B, u64)>,
    // `Sync` because two batches are processed at once, on different threads.
    process: impl Fn(B, u64) -> Vec<Vec<u8>> + Sync,
    // `-v`, so the batch count obeys the same quiet switch as bwa's own progress lines. It is a
    // measurement instrument (see `crates/bwa-cli/tests/batch_count.rs`), so it is on by default.
    verbose: i32,
    // `-K` in bases, echoed next to the count so a log records the setting that produced it.
    k_batch: usize,
) -> anyhow::Result<()> {
    // Rust: everything spawned inside this scope is guaranteed to have finished before the closing
    // brace, and the compiler relies on that guarantee to let those threads borrow local variables
    // directly. Without a scope, each of the reader's and writer's captures would have to be
    // reference-counted and heap-allocated to prove it outlives the thread. The `-> Result` on the
    // closure means the scope as a whole yields the first failure any of them reported.
    std::thread::scope(|scope| -> anyhow::Result<()> {
        // ---- Channels. A couple of batches read-ahead / write-behind is enough to hide I/O
        //      behind compute; deeper queues only cost memory (a batch is ~10M bases) ----
        // Reader -> main. Carries `(batch, base_id)`; blocks the reader once `readahead`
        // batches are queued, which is what bounds resident memory.
        //
        // Rust: `sync_channel(n)` is BOUNDED. Sending into a full queue blocks the sender until the
        // consumer catches up, so the reader cannot run ahead of the aligner and accumulate
        // batches. That blocking is the memory bound: at roughly 10M bases per batch, an unbounded
        // queue would let a fast disk pull an entire whole-genome run into RAM.
        let (_readahead, writebehind) = queue_depths();
        // Main -> writer. Carries one batch's finished SAM bytes as per-record pieces, already in
        // output order; the writer concatenates them into the sink's buffer rather than into a
        // second heap allocation.
        let (sam_tx, sam_rx) = std::sync::mpsc::sync_channel::<Vec<Vec<u8>>>(writebehind);

        // ---- Reader and writer threads. `out` is MOVED into the writer, so it is only ever
        //      touched from one thread ----
        // Join handles. Each yields the thread's `anyhow::Result`, which is the only place a reader
        // or writer error is observed, hence the joins at the end of the scope.
        //
        // Rust: `move` makes each closure take OWNERSHIP of what it mentions. The reader owns its
        // sending half of the batch channel; the writer owns the output sink. Ownership is what
        // makes the single-writer property a compile-time fact rather than a convention: no other
        // thread can name `out` after this line, so no other thread can write to the sink.
        let writer = scope.spawn(move || -> anyhow::Result<()> {
            // Same reason as the reader: the writer allocates (BGZF blocks, htslib buffers) while a
            // stage is running, and `BWA4_STAGE_ALLOC` must not charge that to the stage.
            crate::stage_alloc::set_role(crate::stage_alloc::ROLE_WRITER);
            // Rebound to make the move explicit: the sink now lives on this thread and nowhere else.
            let mut out = out;
            // Rust: a channel receiver can be walked like any other sequence. It yields each value
            // as it arrives, blocking while the queue is empty, and ends by itself once every
            // sender has been dropped. That is why no explicit shutdown message is needed: the loop
            // finishes when the main thread stops sending and releases its half.
            // `BWA4_WRITER_PROBE=1`: report how much of the writer thread's life was spent WRITING
            // versus BLOCKED waiting for a batch. The writer is the pipeline's only serial stage, so
            // "blocked" time is headroom and "writing" time approaching the wall clock means the
            // sink, not the aligner, sets the pace. One `Instant::now()` pair per BATCH (a few dozen
            // per run), never per record, so the measurement cannot pay for itself twice the way a
            // per-call probe does.
            let probe = std::env::var_os("BWA4_WRITER_PROBE").is_some();
            // Seconds spent inside `write_all` and seconds spent inside `recv`. Both stay 0 when the
            // probe is off, and neither is read in that case.
            let (mut writing, mut blocked) = (0f64, 0f64);
            loop {
                let t0 = probe.then(std::time::Instant::now);
                // `recv` returning `Err` is the iterator's stop condition (every sender dropped), so
                // this loop is semantically the `for pieces in sam_rx` it replaces.
                let Ok(pieces) = sam_rx.recv() else { break };
                let t1 = probe.then(std::time::Instant::now);
                // In order, so the concatenation of the pieces is exactly the batch's SAM text.
                // `Output`'s buffering is what turns these back into few, large syscalls.
                for piece in &pieces {
                    out.write_all(piece)?;
                }
                if let (Some(t0), Some(t1)) = (t0, t1) {
                    blocked += t1.duration_since(t0).as_secs_f64();
                    writing += t1.elapsed().as_secs_f64();
                }
            }
            let t_finish = probe.then(std::time::Instant::now);
            out.finish()?;
            if let Some(t) = t_finish {
                eprintln!(
                    "[writer] writing {:.3}s, blocked {:.3}s, finish {:.3}s",
                    writing,
                    blocked,
                    t.elapsed().as_secs_f64()
                );
            }
            Ok(())
        });

        // ---- Main thread: consume batches strictly in order, which is what makes output order
        //      equal input order ----
        // Invariant at the top of each turn: every batch with a smaller `base_id` has already been
        // processed and its bytes handed to the writer, so appending this batch's bytes preserves
        // input order.
        let mut n_batches = 0usize;
        // Spelled as an explicit `recv` loop rather than `for .. in batch_rx` purely so the two
        // BLOCKING operations can be timed apart from the compute between them. `recv` returning
        // `Err` is exactly the iterator's stop condition (the sender was dropped), so the loop is
        // semantically identical to the `for` it replaces.
        // The batch whose `process` is currently running, if any. Holding ONE means two batches are
        // in flight at a time: the one being started and the one still finishing.
        //
        // Why that is worth a thread. A batch walks its stages in sequence (encode, align, rescue,
        // dedup, emit) and they do not fill the pool equally: measured at `-t16` on GRCh38, align
        // keeps it 98.9% busy but rescue is at 86%, dedup at 69% and encode at 29%. Run one batch at
        // a time, those low-occupancy stages are wall clock with half the machine idle, and over ten
        // batches that is seconds. Overlapped, batch N's thin tail runs against batch N+1's align,
        // which is the one stage that can absorb every core.
        //
        // Output order is untouched, and not by luck: the handle is joined and its bytes sent to the
        // writer BEFORE the next batch's handle takes its place, so the writer still sees batches in
        // input order. Only the order in which work is DONE changes, and no batch's result depends
        // on another's (`-K` fixes the boundaries, and the tie-break hash keys on the global read id
        // rather than on anything batch-local).
        //
        // The cost is one more resident batch: records and regions for two batches instead of one,
        // on top of the reader's queue. That is the same currency `queue_depths` spends, and it is
        // spent here because this is where it buys wall clock rather than latency tolerance.
        let mut inflight: Option<std::thread::ScopedJoinHandle<'_, Vec<Vec<u8>>>> = None;
        loop {
            // Blocked here == starved by the reader. Charged to `wait_read`, which is the one
            // number that says whether input (decompress + parse) is the bottleneck.
            let t_wait = std::time::Instant::now();
            let Ok((batch, base_id)) = batch_rx.recv() else {
                break; // reader finished or died; its error surfaces on join below
            };
            stage_time::add(Stage::WaitRead, t_wait.elapsed());
            n_batches += 1;
            stage_time::count_batch();
            crate::stage_alloc::count_batch();
            // Start this batch, THEN retire the previous one: reversing the two lines would be the
            // old serial pipeline with extra threads.
            // `move` so the worker OWNS its batch and id; `process` is borrowed, which is why it
            // has to be `Sync`.
            // `&process` is what moves into the worker, not `process` itself: the closure is shared
            // between the two batches in flight, so it is borrowed and must be `Sync`.
            let process_ref = &process;
            let started = scope.spawn(move || process_ref(batch, base_id));
            // `BWA4_STAGE_ALLOC` tags allocations with a PROCESS-WIDE stage, so two batches in
            // flight would each overwrite the other's tag and every per-stage byte count would be a
            // blend of two stages. Retire this batch before reading the next one while the probe is
            // armed. Output is untouched (batch order is already guaranteed below); only the
            // overlap is lost, so an instrumented run's wall time means nothing, which it did not
            // anyway at four atomics per allocation.
            let started = if crate::stage_alloc::serialize_batches() {
                let buf = started.join().expect("batch worker panicked");
                let t_send = std::time::Instant::now();
                let sent = sam_tx.send(buf);
                stage_time::add(Stage::WaitWrite, t_send.elapsed());
                if sent.is_err() {
                    break; // writer exited; its error surfaces on join below
                }
                continue;
            } else {
                started
            };
            if let Some(prev) = inflight.replace(started) {
                let buf = prev.join().expect("batch worker panicked");
                // Blocked here == throttled by the writer (a slow sink, or BGZF/htslib backpressure).
                let t_send = std::time::Instant::now();
                let sent = sam_tx.send(buf);
                stage_time::add(Stage::WaitWrite, t_send.elapsed());
                if sent.is_err() {
                    break; // writer exited; its error surfaces on join below
                }
            }
        }
        // The last batch has no successor to overlap with, so it is retired here. A send failure
        // needs no branch: the writer's own error is what surfaces, on the join below.
        if let Some(prev) = inflight.take() {
            let buf = prev.join().expect("batch worker panicked");
            let t_send = std::time::Instant::now();
            let _ = sam_tx.send(buf);
            stage_time::add(Stage::WaitWrite, t_send.elapsed());
        }
        drop(sam_tx);
        // Reported because it is the one number that says whether this pipeline did anything: the
        // overlap is batch N+1's read and N-1's write against N's compute, so at a single batch it
        // is structurally inert and the run is not measuring the shipped configuration.
        if verbose >= 3 {
            eprintln!("[M::main_mem] processed {n_batches} batches (-K {k_batch})");
        }

        writer.join().expect("writer thread panicked")?;
        Ok(())
    })
}

/// Hand the index to the operating system instead of freeing it.
///
/// The FM index and the packed reference are about 10 GB on a human genome, spread over three large
/// allocations plus a memory map. Dropping them walks the allocator's structures and returns the
/// pages a region at a time, and that teardown happens after the last SAM byte has already been
/// written, so every millisecond of it is wall clock the user waits for nothing. `mem::forget` skips
/// it; the process is about to exit, and exit reclaims every page at once anyway.
///
/// This is a leak in the strict sense and it is deliberate, so it is worth being precise about why
/// it is sound HERE and would not be in general:
///
/// * Nothing observes these values afterwards. `run` returns immediately after this call, and the
///   output sink was already flushed and closed by the writer thread.
/// * Neither type owns a resource whose release is externally visible: no temporary file to unlink,
///   no lock to hand back. The memory map's descriptor is closed by the kernel at exit like any
///   other.
/// * It is not a growth leak. It happens once, at the end, to values that lived for the whole run.
///
/// If either type ever gains a side effect on drop, this has to go.
fn forget_index(fm: bwa_index::FmIndex, bns: bwa_index::BntSeq) {
    std::mem::forget(fm);
    std::mem::forget(bns);
}

/// Run `bwa-mem4 mem` end to end: build options, load the index, write the SAM header, then stream
/// batches through the align-and-format pipeline (single-end here, paired-end via [`run_pe`]).
///
/// `argv` is the raw process command line, captured in `main` before clap consumed it, and is used
/// verbatim as the `@PG CL:` field so a SAM file records exactly how it was produced.
///
/// # Parameters
///
/// - `args`: the parsed `mem` command line, consumed here. Owns the index prefix, the one or two
///   read files, and every flag.
/// - `argv`: the process's raw arguments including `argv[0]`, joined with single spaces for
///   `@PG CL:`. Deliberately NOT reconstructed from `args`: the tag is meant to record what the
///   user actually typed. Only ever read, never parsed.
///
/// # Returns
///
/// `Ok(())` after the last SAM record has been written and the sink finalized. Errors: an
/// unreadable index, an unreadable `-H` file, a malformed `-I`/`-O`/`-E`/`-L`/`-h` argument, or any
/// I/O failure in the pipeline.
pub fn run(args: MemArgs, argv: &[String]) -> anyhow::Result<()> {
    // Wall-clock origin for the end-of-run traffic dump only; not on any hot path.
    let t_run = std::time::Instant::now();
    // Options we parse for CLI compatibility but do not yet honour. Failing loudly is the point: an
    // aligner that accepts `-a` and silently emits only primaries hands back a result the user
    // believes is something it is not. Refuse rather than mislead.
    // Each tuple is (was the flag given, its spelling for the message, what it would have done).
    // Currently EMPTY: every option this port accepts is now honoured. The machinery is kept
    // because it is the place to re-add an entry the moment a new flag is parsed ahead of being
    // implemented.
    let unimplemented: &[(bool, &str, &str)] = &[];
    if let Some((_, flag, what)) = unimplemented.iter().find(|(given, _, _)| *given) {
        anyhow::bail!(
            "{flag} is not implemented yet ({what}). bwa-mem4 parses it for CLI compatibility but \
             would ignore it, so it refuses rather than emit output that silently differs from \
             bwa-mem2."
        );
    }

    // ================= Long-read routing, before any bwa work =================
    //
    // `-x pacbio|pbref|ont2d` leaves this file entirely: bwa's long-read presets are a short-read
    // design retuned, and bwa-mem2's own code tells the user not to use them. See `cmd_longread` for
    // the reasoning, the preset mapping and what it does to the output contract. `-x intractg` is
    // NOT routed and continues below on the bwa-identical path.
    //
    // Placed before `build_opt` because none of bwa's options apply to a different mapper; the only
    // thing consumed here is `-o` (where the SAM goes) and the read paths.
    if let Some(mode) = args.read_type.as_deref() {
        if let Some(preset) = crate::cmd_longread::preset_for(mode) {
            let mut reads = vec![args.reads.clone()];
            if let Some(r2) = args.reads2.clone() {
                reads.push(r2);
            }
            // Same sink choice as the short-read path: a file when `-o` was given, else stdout.
            let mut out: Box<dyn std::io::Write> = match &args.output {
                Some(path) => Box::new(std::io::BufWriter::new(
                    std::fs::File::create(path)
                        .map_err(|e| anyhow::anyhow!("creating {}: {e}", path.display()))?,
                )),
                None => Box::new(std::io::BufWriter::new(std::io::stdout().lock())),
            };
            // The long-read path runs on the rayon pool too, so `-t` must be honoured here: the
            // pool setup further down is never reached on this branch. Same fixed-size pool, and
            // the same reason it is fixed: nothing about the mapping depends on how many workers
            // there are, so the thread count is a speed knob only.
            rayon::ThreadPoolBuilder::new()
                .num_threads(args.threads.max(1) as usize)
                .build_global()
                .ok();
            crate::cmd_longread::run(preset, mode, &args.index_prefix, &reads, &mut out, argv)?;
            out.flush()?;
            return Ok(());
        }
    }

    // ================= Options and thread pool =================

    // `-C`: process-wide, like bwa's `aux.copy_comment`.
    bwa_core::rg::set_copy_comment(args.copy_comment);

    // Every tunable, final and immutable from here on: shared by reference with every worker.
    let opt = build_opt(&args)?;
    // `-t` clamped to >= 1, as a usize for the pool builders. Same value as `opt.n_threads`.
    let n_threads = args.threads.max(1) as usize;

    // Worker count for the POOL ONLY, capped to the Performance cores on Apple Silicon.
    //
    // READ THIS BEFORE MOVING THE CAP ANYWHERE ELSE. It must not touch `args.threads` or
    // `opt.n_threads`, because `-K` defaults to `opt.chunk_size * args.threads` a few lines below
    // (bwa's `aux.task_size = opt->chunk_size * opt->n_threads`, fastmap.cpp:964). Capping the
    // thread count itself would therefore shrink the default batch, move the batch boundaries, and
    // change the paired-end output, since the insert-size model is re-estimated per batch. A
    // "threading only" change would then silently break byte-identity. The pool size, by contrast,
    // is invisible in the output: order and global read ids do not depend on it.
    //
    // Why cap at all: on an M4 Max (12 P + 4 E) the four E cores measurably buy nothing (`-t16`
    // 6.10 s wall against `-t12` 6.17 s, inside the spread of three repetitions) while costing 10%
    // more CPU, because rayon splits work evenly and an E core here is several times slower, so its
    // chunk straggles. See `bwa_core::cpu` for the measurements. macOS offers no way to forbid a
    // core -- affinity is a no-op on arm64 and QoS is only a hint -- so not creating the extra
    // workers is the only lever that works.
    //
    // MEASURED AGAIN (2026-07-29) ON THE REAL WORKLOAD, AND THE CAP IS NOW OFF BY DEFAULT.
    // On 1M real GIAB pairs, gzipped, paired-end, `-t16`, default `-K`, alternating, 3 reps:
    //
    //   capped to 12 P cores : 24.25 / 26.31 / 26.39 s   (mean 25.65)
    //   all 16 cores         : 23.00 / 23.09 / 23.82 s   (mean 23.30)   -9.2%, 3 of 3
    //   fg-labs/bwa-mem3     : 25.38 / 24.74 / 27.37 s   (mean 25.83)
    //
    // The cap's original claim ("no measurable throughput, ~8% more CPU") holds for CPU time and is
    // WRONG for wall time here: uncapped burns more CPU (281-324s against 260-290s) and still
    // finishes 9% sooner, because a slow core doing some of the work still beats an idle one. The
    // measurement that justified the cap was on the small simulated benchmark, where mate rescue is
    // nearly cold and the E-core straggler dominates a short parallel region; on real paired-end
    // data rescue is ~49% of wall and there is plenty of work to hide the stragglers behind.
    //
    // It also decided the head-to-head: capped we were at parity with the fork at `-t16`, uncapped
    // we are 1.10x ahead (paired ratios 1.103 / 1.071 / 1.149).
    //
    // `BWA4_PCORE_CAP=1` restores the old behaviour, which is how the effect stays measurable and
    // how anyone hitting the simulated-benchmark regime can get it back. Nothing here can change
    // output: the pool size is independent of `-K`, which is what fixes batch boundaries.
    let pool_threads = match bwa_core::cpu::performance_core_count() {
        Some(p) if n_threads > p && std::env::var_os("BWA4_PCORE_CAP").is_some() => {
            // `-v` default is 3 (bwa's `bwa_verbose`), so this prints unless the user quietened it.
            if args.verbose.unwrap_or(3) >= 3 {
                eprintln!(
                    "[M::main_mem] BWA4_PCORE_CAP set: -t {n_threads} exceeds the {p} performance \
                     cores; running {p} workers."
                );
            }
            p
        }
        _ => n_threads,
    };
    // Fixed-size rayon pool. Output order and global read ids are independent of thread count, so
    // byte-identity holds at any `-t` once `-K` fixes the batch boundaries.
    rayon::ThreadPoolBuilder::new()
        .num_threads(pool_threads)
        .build_global()
        .ok();
    // Batch size in INPUT BASES, not reads. Default `chunk_size * -t` (10M per thread), mirroring
    // bwa's `aux.task_size`; `-K` overrides it. Load-bearing for paired-end reproducibility: the
    // insert-size model is re-estimated per batch, so the boundary this fixes is visible in the
    // output. Clamped to >= 1 so a zero or negative `-K` cannot stall the reader.
    let k_batch = args
        .k_batch
        .unwrap_or(opt.chunk_size * i64::from(args.threads))
        .max(1) as usize;

    // ================= Start reading before loading the index =================
    //
    // The reader opens the FASTQ, inflates it if gzipped and parses records; at the default `-K` on
    // a 16-thread run that first batch is 160M bases, which is on the order of a second of work. The
    // index load is another ~1.8 s. Started here rather than inside the pipeline, the two overlap
    // and the run pays for the longer of them instead of their sum.
    //
    // The channel is bounded at the same readahead depth as before, so the reader still cannot
    // accumulate batches while the index loads: it fills the queue and blocks. Output is unaffected,
    // since the batches, their order and their base ids are identical either way.
    // Decoder budget for gzipped input, set from `-t` before the first file is opened. Halved per
    // file inside, since a paired-end run opens two. Spending threads here is free where it counts:
    // during the first batch no worker has anything to do yet.
    #[cfg(feature = "parallel-gzip")]
    bwa_io::fastq::set_gzip_threads(pool_threads);

    let paired = args.reads2.is_some() || args.smart_pairing;
    let (pe_batch_rx, se_batch_rx, reader) = if paired {
        let (rx, h) = spawn_reader(pe_read_batches(
            args.reads.clone(),
            args.reads2.clone(),
            k_batch,
        ));
        (Some(rx), None, h)
    } else {
        let (rx, h) = spawn_reader(se_read_batches(args.reads.clone(), k_batch));
        (None, Some(rx), h)
    };

    // ================= Load the index =================
    //
    // The two halves of a bwa index: `FmIndex` is the FMD index used for seeding (exact-match
    // lookup), `BntSeq` is the contig dictionary plus the 2-bit packed reference used for
    // extension and for turning a concatenated-genome offset back into (contig, position).
    // Both are loaded once, live for the whole run, and are shared immutably with every worker.
    // Together they are several GB for a human genome (memory-mapped where possible).
    let fm = FmIndex::load(&args.index_prefix)?;
    let mut bns = BntSeq::load(&args.index_prefix)?;
    // `-j`: clear the ALT flags the loader just read from `<prefix>.alt`, so every contig counts as
    // primary assembly. Done here, after loading, exactly as `fastmap.cpp:907` does it.
    if args.ignore_alt {
        bns.ignore_alt();
    }
    // bwa prints this at load time (`bwa.cpp:419`). Kept behind the same verbosity gate as bwa's
    // other progress chatter: it goes to stderr and never touches the SAM bytes.
    if args.verbose.unwrap_or(3) >= 3 && bns.n_alt() > 0 {
        eprintln!("[M::main_mem] read {} ALT contigs", bns.n_alt());
    }
    let bns = bns;
    // Everything resident from here to the end of the run is the index and nothing else, so this is
    // the line that lets `BWA4_STAGE_ALLOC` report a per-batch peak instead of 10 GB of genome on
    // every row. No-op unless that probe is armed.
    crate::stage_alloc::mark_baseline();
    // One `@SQ` line per contig, in index order. That order comes from the FASTA and must be
    // preserved: downstream tools treat @SQ order as the coordinate sort order.
    //
    // Known gap vs the C: `bwa_print_sam_hdr` also appends `\tAH:*` to an ALT contig's @SQ line
    // (`bwa.cpp:530`). `Contig` here has no `is_alt` field, so that suffix is never emitted. It is
    // unreachable in practice because no `.alt` file is read at all (see `-j`).
    // The `@SQ` payload: (name, length in bases) per contig, index order preserved. Used only to
    // write the header, then dropped.
    let sqs: Vec<SqRecord> = bns
        .contigs
        .iter()
        .map(|contig| SqRecord {
            name: contig.name.clone(),
            len: i64::from(contig.len),
            is_alt: contig.is_alt,
        })
        .collect();

    // ================= Assemble the SAM header =================
    //
    // `-H` first, then `-R` appended to it: `main_mem` builds hdr_line during getopt and only after
    // the loop does `hdr_line = bwa_insert_header(rg_line, hdr_line)`, so @RG lands last.
    // Extra header lines in emission order: the `-H` lines first, then the `-R` @RG line. Each entry
    // is one complete, already-escape-expanded header line without its newline. Empty when neither
    // option was given.
    let mut hdr_parts: Vec<String> = Vec::new();
    if let Some(header_arg) = &args.header_insert {
        if let Some(rest) = header_arg.strip_prefix('@') {
            hdr_parts.push(bwa_core::rg::escape(&format!("@{rest}")));
        } else {
            // Not starting with '@': bwa treats it as a FILE of header lines.
            // Whole file slurped; non-`@` lines are skipped below rather than rejected.
            let text = std::fs::read_to_string(header_arg)
                .map_err(|e| anyhow::anyhow!("-H: cannot read '{header_arg}': {e}"))?;
            for line in text.lines() {
                if line.starts_with('@') {
                    hdr_parts.push(bwa_core::rg::escape(line));
                }
            }
        }
    }
    if let Some(rg_arg) = &args.read_group {
        // Installs the parsed ID process-wide as a side effect, and returns the header line.
        // The escape-expanded `@RG` line; the ID it also installed is what every record's `RG:Z:`
        // tag will carry, so this call must happen before any record is written.
        let rg_header_line =
            bwa_core::rg::set_rg(rg_arg).map_err(|e| anyhow::anyhow!("-R: {e}"))?;
        hdr_parts.push(rg_header_line);
    }
    // The extra header block as one newline-joined string, or `None` for "no extra lines" (which is
    // what `write_header` needs to distinguish, since an empty string would emit a blank line).
    let hdr_lines = if hdr_parts.is_empty() {
        None
    } else {
        Some(hdr_parts.join("\n"))
    };

    // `-I` fixes the insert-size distribution instead of inferring it per batch.
    // `Some` = use these four orientation stats for every batch, `None` = call `mem_pestat` per
    // batch. Single-end runs ignore it entirely.
    let pes0 = match &args.insert_size {
        Some(s) => Some(parse_insert_size(s)?),
        None => None,
    };

    // ================= Open the sink and write the header =================
    // The sink. Written to directly for the header here, then MOVED into the pipeline's writer
    // thread, which is the only thread that touches it afterwards.
    let mut out = Output::open(args.output.as_deref(), n_threads, args.reference.as_deref())?;
    // The raw command line for `@PG CL:`. Single-space joined, so a quoted argument containing
    // spaces is not re-quoted; that matches what bwa writes.
    //
    // DIVERGENCE FROM bwa-mem2, deliberate: tabs are escaped back to the two characters `\t`.
    //
    // bwa-mem2 writes the argument verbatim, so `-R "$(printf '@RG\tID:x')"` puts a REAL tab
    // inside `CL:`, which is a SAM field separator. The `@PG` line then has seven fields instead of
    // four and carries two `ID:` tags, one from the header and one from the read group. That is an
    // invalid header: lenient parsers shrug, strict ones (noodles) reject the file. Reported
    // upstream as bwa-mem2#293, still open.
    //
    // Escaping rather than stripping, because `\t` is exactly the spelling bwa itself ACCEPTS on
    // the `-R` command line, so the field stays a faithful record of what the user typed and could
    // be pasted back. This costs no byte-identity: `@PG` is excluded from every parity gate in this
    // project precisely because it legitimately differs between the two tools.
    let command_line = argv.join(" ").replace('\t', "\\t");
    sam::write_header(
        &mut out,
        &sqs,
        hdr_lines.as_deref(),
        "bwa-mem4",
        "bwa-mem4",
        env!("CARGO_PKG_VERSION"),
        &command_line,
    )?;

    // ================= Dispatch: paired-end leaves here, single-end continues below =================
    //
    // Paired-end when a second file is given, or when `-p` says the one file is interleaved.
    if args.reads2.is_some() || args.smart_pairing {
        // Cloned so the borrow of `args` ends before `args.reads` is borrowed in the same call.
        // `None` here means `-p`: one interleaved file.
        run_pe(
            &fm,
            &bns,
            &opt,
            pe_batch_rx.expect("paired branch always has the paired receiver"),
            k_batch,
            out,
            pes0,
            args.verbose.unwrap_or(3),
        )?;
        reader.join().expect("reader thread panicked")?;
        bwa_neon::matesw::cells::dump();
        bwa_neon::batched::extend_shape::dump();
        bwa_mem::across::align_split::dump();
        bwa_mem::seed_stats_dump();
        bwa_mem::pe::rescue_rounds::dump();
        bwa_mem::pe::rescue_rounds::dump_clusters();
        bwa_mem::primary::dedup_shape::dump();
        bwa_mem::pe::anchor_spread::dump();
        bwa_chain::chain_time::dump();
        bwa_index::traffic::dump(t_run.elapsed().as_secs_f64());
        // Last, because it is the widest view: the others break down one stage, this one accounts
        // for all of them. Runs on the main thread, where the stage accumulators live.
        stage_time::dump(t_run.elapsed());
        stage_time::barrier::dump();
        // Where the BYTES went, next to where the wall clock went. The line above was duplicated,
        // which printed the occupancy table twice.
        crate::stage_alloc::dump();
        forget_index(fm, bns);
        return Ok(());
    }

    // ================= Single-end pipeline =================
    //
    // The reader for this path was already started above, before the index load; `se_batch_rx` is
    // its receiving half.
    let batch_rx = se_batch_rx.expect("single-end branch always has the single-end receiver");

    // Main: seed -> chain -> BATCHED seed extension across the whole read batch (NEON backend),
    // mirroring bwa-mem2's mem_chain2aln_across_reads_V2. Chunked so extension parallelizes; each
    // read's regions are independent of chunk composition, so output stays byte-identical at any
    // thread count once -K fixes the batch boundaries.
    // `batch` is one `-K` worth of reads in input order; `base_id` is the global id of its first
    // read (see the reader above). Returns the batch's complete SAM text.
    let process = |batch: Vec<Record>, base_id: u64| -> Vec<Vec<u8>> {
        // ASCII bases -> nt4 codes (A=0, C=1, G=2, T=3, anything else 4), once per read.
        // Parallel to `batch`: `all_codes[i]` is `batch[i]`'s sequence, same length.
        //
        // Run on the pool, as bwa-mem2 does (it encodes inside `mem_kernel1_core`, under `kt_for`).
        // `dna::nt4` is a pure 256-entry table lookup, so `all_codes[i]` depends on `batch[i]` and
        // nothing else, and `par_iter` is INDEXED so element order is preserved: byte-identical.
        let all_codes: Vec<Vec<u8>> = stage_time::measure(Stage::Encode, || {
            batch
                .par_iter()
                .map(|rec| rec.seq().iter().map(|&base| dna::nt4(base)).collect())
                .collect()
        });
        // The denominator of the bytes-per-base column issue #25 is written in. One sum per batch,
        // and only when `BWA4_STAGE_ALLOC` is armed.
        if crate::stage_alloc::enabled() {
            crate::stage_alloc::note_bases(all_codes.iter().map(|c| c.len() as u64).sum());
        }
        // Per-read candidate alignments BEFORE dedup and primary marking, also parallel to `batch`.
        let regs_all =
            stage_time::measure(Stage::Align, || batched_regs(&fm, &bns, &opt, &all_codes));
        // Move each read's regions out of `regs_all` (consumed by `finish_se`) instead of cloning:
        // `into_par_iter` yields the owned `Vec<MemAlnReg>`, dropping a per-read Vec allocation+copy.
        // One entry per read: that read's finished SAM records (possibly several, possibly one
        // unmapped record, never empty). rayon's indexed parallel iterators preserve read order.
        let per_read_sam: Vec<Vec<u8>> = stage_time::measure(Stage::SamEmit, || {
            batch
                .par_iter()
                .zip(all_codes.par_iter())
                .zip(regs_all.into_par_iter())
                .enumerate()
                .map(|(read_in_batch, ((rec, codes), regs_pre))| {
                    finish_se(
                        &fm,
                        &bns,
                        &opt,
                        rec,
                        codes,
                        regs_pre,
                        base_id + read_in_batch as u64,
                    )
                })
                .collect()
        });
        // Handed to the writer as pieces, in read order (the indexed parallel map above preserved
        // it). No concatenation: see `run_pipeline`'s docs for why that buffer is not built.
        per_read_sam
    };

    run_pipeline(out, batch_rx, process, args.verbose.unwrap_or(3), k_batch)?;
    reader.join().expect("reader thread panicked")?;
    bwa_chain::chain_time::dump();
    bwa_index::traffic::dump(t_run.elapsed().as_secs_f64());
    // Last, because it is the widest view: the others break down one stage, this one accounts for
    // all of them. Runs on the main thread, where the stage accumulators live.
    stage_time::dump(t_run.elapsed());
    crate::stage_alloc::dump();
    forget_index(fm, bns);
    Ok(())
}

/// Seed + chain + batched extension for a whole read batch, returning each read's pre-dedup regions
/// (byte-identical to `align_read` per read). Chunked across the rayon pool so extension batches run
/// in parallel; per-read results are independent of the chunking.
///
/// # Parameters
///
/// - `fm`: the FM index, for seeding. Immutable and shared by every worker.
/// - `bns`: contig dictionary plus the packed reference, for extension and coordinate translation.
/// - `opt`: the fully-derived options; read-only.
/// - `codes`: one nt4-coded sequence per read (A=0, C=1, G=2, T=3, else 4). For paired-end this is
///   the interleaved c1,c2,c1,c2,... list, so its length is twice the pair count.
/// - `backend`: which seed-extension implementation to call. Byte-identical either way.
///
/// # Returns
///
/// One `Vec<MemAlnReg>` per entry of `codes`, in the same order. These are PRE-dedup and unmarked:
/// the caller still owes `mem_sort_dedup_patch` and `mem_mark_primary_se`.
fn batched_regs(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    codes: &[Vec<u8>],
) -> Vec<Vec<MemAlnReg>> {
    // One chunk per worker, so every thread gets a slice big enough to fill the SIMD lanes.
    // Size of the rayon pool, i.e. effectively `-t`. Clamped so the division below cannot divide
    // by zero.
    let worker_count = rayon::current_num_threads().max(1);
    // Reads per parallel chunk. `div_ceil(worker_count)` alone makes each chunk `batch/-t` reads, so
    // every worker materialises its whole share of the batch's seeding intermediates (SMEMs, resolved
    // SA positions, chains, pre-dedup regions) at once and peak RSS scales with `-K` (= 10M*-t by
    // default). We instead size the chunk so the intermediates held CONCURRENTLY are a fixed budget
    // INDEPENDENT of `-t`: with `-t` chunks in flight, `chunk = INFLIGHT_READS / -t` gives
    // `-t * chunk = INFLIGHT_READS` reads resident regardless of thread count. A floor keeps each
    // lockstep batch wide enough to hide FM-index latency (so very high `-t` trades the flat-RAM
    // property for lockstep width rather than the reverse), and we never chunk larger than one
    // per-worker share. Scheduling only -- the per-read regions do not depend on the chunk size.
    const INFLIGHT_READS: usize = 16384; // total reads' intermediates resident at once, all workers
    const MIN_CHUNK: usize = 512; // lockstep floor: enough reads in flight to hide FM latency
    let budget_chunk = (INFLIGHT_READS / worker_count).max(MIN_CHUNK);
    let reads_per_chunk = codes.len().div_ceil(worker_count).min(budget_chunk).max(1);
    crate::stage_time::barrier::region(crate::stage_time::Stage::Align, || {
        codes
            .par_chunks(reads_per_chunk)
            .flat_map(|chunk| {
                crate::stage_time::barrier::worker(|| {
                    align_reads_batched(fm, bns, opt, chunk, &NeonBackend)
                })
            })
            .collect()
    })
}

/// Env-gated (`BWA4_DUMP_REGS`) region dump; cached, since `finish_se` runs per read.
///
/// `BWA4_DUMP_REGS`: set to ANY value (its content is never read, only its presence) to print each
/// read's candidate regions to stderr before and after dedup/primary-marking. Default off. A
/// debugging aid for parity work against the oracle: it writes to stderr only, so it cannot change
/// the SAM bytes, but it is very slow and unordered under `-t > 1`.
///
/// # Returns
///
/// Whether the dump is on. Read once per process and cached, because this is called for every read.
fn dump_regs_enabled() -> bool {
    // Cached decision. `OnceLock` rather than a plain static so the env read happens exactly once
    // even when several rayon workers reach it simultaneously.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BWA4_DUMP_REGS").is_some())
}

/// Deduplicate + primary-mark a read's batched regions, then format its SAM record. Pure (no shared
/// state beyond the immutable index/options), so it is safe across rayon workers.
///
/// # Parameters
///
/// - `fm`, `bns`, `opt`: the shared immutable index halves and options.
/// - `rec`: the original read, for QNAME, SEQ, QUAL and the `-C` comment.
/// - `codes`: `rec.seq` in nt4 coding, same length; the alignment code works on this, not on ASCII.
/// - `regs_pre`: this read's pre-dedup candidate regions from [`batched_regs`], CONSUMED here
///   (`mem_sort_dedup_patch` takes ownership) to avoid a per-read clone.
/// - `read_id`: the read's global 0-based index across the whole input, not its index within the
///   batch. Load-bearing: `mem_mark_primary_se` hashes it to break score ties, so a per-batch id
///   would change the output.
///
/// # Returns
///
/// The read's SAM text: one record per surviving alignment, or a single unmapped (FLAG 4) record if
/// none survived. Never empty, and always newline-terminated by the `sam` writers.
fn finish_se(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    rec: &Record,
    codes: &[u8],
    regs_pre: Vec<MemAlnReg>,
    read_id: u64,
) -> Vec<u8> {
    // ---- Deduplicate overlapping candidate regions, then mark which is primary ----
    if dump_regs_enabled() {
        eprintln!("=== read {} ===", rec.name());
        bwa_mem::dump_regs(bns, "pre-dedup", &regs_pre);
    }
    // The read's regions after redundant ones have been merged away, sorted, and then annotated by
    // `mem_mark_primary_se` with `secondary` (-1 = unshadowed, else the index of the shadower).
    // After marking, `regs[0]` is the highest-scoring primary.
    let mut regs = mem_sort_dedup_patch(fm, opt, codes, regs_pre);
    // `bwamem.cpp:1161`: re-stamp is_alt from each region's own contig, after the dedup and
    // before any primary marking reads it.
    bwa_mem::stamp_is_alt(bns, &mut regs);
    mem_mark_primary_se(opt, &mut regs, read_id);
    if dump_regs_enabled() {
        bwa_mem::dump_regs(bns, "post-dedup+mark", &regs);
    }

    // ---- Alternative-hit tags. After marking, regs[0] is the highest-scoring primary region.
    //      `-a` (MEM_F_ALL) emits the shadowed regions as their own secondary records, so bwa skips
    //      the XA tag entirely in that mode: the same information cannot be in both places ----
    // `-a`. Read several times below, so it is hoisted once.
    let output_all = opt.flag & flags::ALL != 0;
    // Parallel to `regs`: the `XA:Z` value for each region, or `None` where there is none. All-`None`
    // under `-a`, since the shadowed hits become their own records instead.
    let xa_per_reg = if output_all {
        vec![None; regs.len()]
    } else {
        mem_gen_alt(fm, bns, opt, &regs, codes.len() as i32, codes)
    };

    // `mem_reg2sam`: emit every region clearing -T. Without `-a`, a region shadowed by a better
    // overlapping one (`secondary >= 0`) is skipped and surfaces in the primary's XA:Z instead;
    // with `-a` it is emitted as a FLAG 0x100 record (MAPQ 0, no XS, SEQ/QUAL `*`), subject to the
    // drop-ratio test. Among the *primary* survivors the first is the primary record and the rest
    // are supplementary (0x800, or 0x100 under `-M`).
    // The records to emit, accumulated in region order. Invariant at the top of each loop turn:
    // `alns` holds exactly the regions before `reg_idx` that passed the filters, so `alns[0]` (once
    // non-empty) is this read's primary record, and `alns.is_empty()` is the test for "the region
    // about to be pushed is the primary". Both facts are used inside the loop.
    let mut alns: Vec<EmitAln> = Vec::new();
    for (reg_idx, reg) in regs.iter().enumerate() {
        // ---- Filter: below `-T`, or shadowed and not being emitted separately ----
        if reg.score < opt.t {
            continue;
        }
        // `secondary >= 0` means "shadowed by region N"; -1 means this region is unshadowed.
        if reg.secondary >= 0 && (reg.is_alt || !output_all) {
            continue;
        }
        // `p->score < a->a[p->secondary].score * opt->drop_ratio`, in C's single precision.
        if reg.secondary >= 0 {
            // Score of the region that shadows this one; the emit floor is that times `-D`.
            let shadowing_score = regs[reg.secondary as usize].score;
            if (reg.score as f32) < shadowing_score as f32 * opt.drop_ratio {
                continue;
            }
        }

        // ---- Build the alignment (exact CIGAR + NM/MD) and its MAPQ ----
        // Reaching here with `secondary >= 0` implies `-a` (the filter above dropped it otherwise),
        // so this is exactly "emit as a FLAG 0x100 record".
        let is_secondary = reg.secondary >= 0;
        // The region turned into a real alignment: exact CIGAR, NM, MD, AS and the sub-optimal
        // score for XS. Mutable only so the secondary branch can clear `sub`.
        let mut aln = reg2aln(fm, bns, opt, codes.len() as i32, codes, reg);
        // `mem_reg2aln`: a secondary gets MAPQ 0 and does not advertise a sub-optimal score.
        // MAPQ for SAM column 5, 0..=60. Mutable because the supplementary cap below may lower it.
        let mut mapq = if is_secondary {
            aln.sub = -1;
            0
        } else {
            mem_approx_mapq_se(opt, reg)
        };
        // Non-first *primary* hits are supplementary. A secondary never is.
        // `!alns.is_empty()` is precisely "a record has already been emitted for this read", per the
        // loop invariant above; it decides FLAG 0x800 (or 0x100 under `-M`) and hard clipping.
        let is_supplementary = !is_secondary && !alns.is_empty();
        // bwa caps a later record's MAPQ at the first record's, unless `-5`/`-q` asked to keep it.
        if !alns.is_empty()
            && opt.flag & flags::KEEP_SUPP_MAPQ == 0
            && !reg.is_alt
            && mapq > alns[0].mapq
        {
            mapq = alns[0].mapq;
        }
        alns.push(EmitAln {
            aln,
            mapq,
            is_alt: reg.is_alt,
            xa: xa_per_reg[reg_idx].clone(),
            secondary: is_secondary,
            sup: is_supplementary,
        });
    }

    // ---- Emit: an unmapped record when nothing survived, else one record per survivor ----
    // This read's SAM text, the function's return value.
    let mut buf = Vec::new();
    if alns.is_empty() {
        sam::write_unmapped(&mut buf, rec.name(), rec.seq(), rec.qual(), rec.comment())
            .expect("write to Vec");
        return buf;
    }
    for which in 0..alns.len() {
        write_aln_se(&mut buf, bns, opt, rec, &alns, which);
    }
    buf
}

/// One emitted alignment plus the per-record state `mem_aln2sam` needs (`mem_aln_t` + its MAPQ,
/// which we compute outside `reg2aln`).
struct EmitAln {
    /// The alignment itself: rid, pos (0-based), strand, cigar (`oplen << 4 | op` in bwa's `MIDSH`
    /// numbering), NM, MD, AS and the sub-optimal score. Produced by `reg2aln` and not modified
    /// afterwards, except that a secondary has had `sub` forced to -1.
    aln: MemAln,
    /// SAM column 5, 0..=60. Computed outside `reg2aln` because bwa computes it outside
    /// `mem_reg2aln` too: 0 for a secondary, `mem_approx_mapq_se` otherwise, then possibly capped
    /// at the primary's MAPQ unless `-5`/`-q`. Also printed inside other records' `SA:Z`.
    mapq: u32,
    /// Whether the region landed on an ALT contig. Currently ALWAYS false in this port: no `.alt`
    /// file is read, so no contig is ever marked ALT (see `-j`). Kept because it gates hard
    /// clipping and the MAPQ cap exactly as in the C.
    is_alt: bool,
    /// The `XA:Z` value for this record, or `None` when there is none (always `None` under `-a`).
    /// Cloned in from `xa_per_reg`, already formatted as bwa's `;`-separated hit list.
    xa: Option<String>,
    /// Shadowed region emitted under `-a`: FLAG 0x100, SEQ/QUAL `*`, never in anyone's SA:Z.
    secondary: bool,
    /// A later *primary* hit: supplementary (0x800, or 0x10000 -> 0x100 under `-M`).
    sup: bool,
}

/// `mem_aln2sam` for one of a read's alignments. `which` indexes `alns`: 0 is the primary, any other
/// is a supplementary record (FLAG 0x800), which hard-clips and carries only its own slice of
/// SEQ/QUAL so the read's bases are stored exactly once across the records.
///
/// # Parameters
///
/// - `buf`: append-only byte sink for this read's records; the caller owns ordering.
/// - `bns`: contig dictionary, used only to turn `rid` into an RNAME string.
/// - `opt`: read for `flag` bits `SOFTCLIP` and `NO_MULTI` only.
/// - `rec`: the original read, needed for QNAME and for the untouched forward SEQ/QUAL.
/// - `alns`: ALL of this read's emitted alignments. Passed whole, not just `alns[which]`, because
///   `SA:Z` has to list the others; this is `mem_aln2sam`'s `(n, list, which)` triple.
/// - `which`: index into `alns`. 0 is the primary; the rest are supplementary or secondary.
///
/// # The three SAM fields with non-obvious arithmetic
///
/// **FLAG.** Built in two steps. First the bits are accumulated, including the pseudo-bit `0x10000`
/// for "supplementary, but `-M` says report it as secondary". Then the emitted value is
/// `(flag & 0xffff) | (flag & 0x10000 ? 0x100 : 0)`. The detour exists because `-M` must turn a
/// supplementary into a *secondary* without ever letting 0x800 and 0x100 both be set, and without
/// 0x10000 (not a real SAM bit) escaping into the file. Bits set here: 0x10 reverse strand, 0x100
/// secondary, 0x800 supplementary. The pairing bits (0x1/0x2/0x40/0x80) are the PE path's job.
///
/// **SEQ/QUAL.** A secondary record writes `*` for both: its bases are already on the primary. A
/// supplementary writes only the `[qb, qe)` slice its CIGAR actually covers, since the hard-clipped
/// ends live on another record. `qb`/`qe` are computed on the FORWARD read and only then
/// reverse-complemented, which is why the reverse-strand branch swaps which cigar end trims which.
///
/// **Tag order.** Fixed, and byte-identity depends on it: `NM MD AS XS RG SA XA <comment>`.
fn write_aln_se(
    buf: &mut Vec<u8>,
    bns: &BntSeq,
    opt: &MemOpt,
    rec: &Record,
    alns: &[EmitAln],
    which: usize,
) {
    // The record being written, and its alignment. Everything else in `alns` is only consulted for
    // the `SA:Z` tag.
    let emit = &alns[which];
    let aln = &emit.aln;
    // `-Y`: soft-clip supplementary records instead of hard-clipping them.
    let softclip = opt.flag & flags::SOFTCLIP != 0;
    // Only a supplementary record hard-clips; a secondary carries `*` for SEQ/QUAL anyway.
    // True means "this record stores only part of the read", which is what drives the `qb`/`qe`
    // slicing below. The empty-cigar guard is for an alignment that has no cigar to read clips off.
    let clip_ends = emit.sup && !softclip && !emit.is_alt && !aln.cigar.is_empty();

    // ---- FLAG (SAM column 2) ----
    // Accumulator, seeded with whatever `reg2aln` already set (the pairing bits in the PE path;
    // nothing here). May transiently hold the pseudo-bit 0x10000, which the rebind below strips.
    let mut flag = aln.flag;
    if aln.is_rev {
        flag |= FLAG_REVERSE;
    }
    if emit.secondary {
        flag |= FLAG_SECONDARY;
    }
    if emit.sup {
        flag |= if opt.flag & flags::NO_MULTI != 0 {
            FLAG_SUPP_REPORTED_AS_SECONDARY
        } else {
            FLAG_SUPPLEMENTARY
        };
    }
    // bwa prints `(flag & 0xffff) | (flag & 0x10000 ? 0x100 : 0)`: `-M` reports a supplementary as
    // secondary without letting 0x10000 leak into the emitted FLAG.
    // Shadows the accumulator: from here on `flag` is the exact value printed in column 2.
    let flag = (flag & FLAG_SAM_BITS)
        | if flag & FLAG_SUPP_REPORTED_AS_SECONDARY != 0 {
            FLAG_SECONDARY
        } else {
            0
        };

    // ---- SEQ/QUAL slice bounds. `qb`/`qe` are bwa's query begin/end: the half-open range of the
    //      FORWARD read this record actually stores ----
    //
    // Hard-clipped ends are not in this record's SEQ/QUAL. bwa reads the clip lengths off both
    // cigar ends (which coincide for a 1-op cigar) and maps them onto the *forward* read.
    // Half-open `[qb, qe)` over the FORWARD read, in bases. Starts as the whole read (the primary,
    // and any soft-clipping record, stores all of it) and is narrowed below only when hard clipping
    // applies. Invariant: `0 <= qb <= qe <= rec.seq.len()`, and it indexes `rec.seq`/`rec.qual`
    // BEFORE any reverse-complementing.
    let (mut qb, mut qe) = (0usize, rec.seq().len());
    if clip_ends {
        // The two end ops. For a single-op cigar these are the same entry, and the two `if`s below
        // would both fire on it; that mirrors the C.
        let (first_op, last_op) = (aln.cigar[0], aln.cigar[aln.cigar.len() - 1]);
        // Cigar encoding is `oplen << 4 | op`. Careful: bwa uses its OWN 5-op table `MIDSH`, so
        // 3 = S and 4 = H (`bwamem.h:49`: "op to integer mapping: MIDSH=>01234", and the decoder in
        // `bwa_mem::cigar::cigar_string` indexes the same 5-char table). This is NOT the SAM/BAM
        // spec's `MIDNSH` ordering, where 3 would be N and 4 S. Reading this line against the spec
        // table instead of bwa's makes it look off by one; it is not. See the consts at the top.
        // "Is this packed cigar entry a clip of either kind?" Takes the whole entry, not just the
        // op nibble, and masks internally.
        let is_clip = |op: u32| {
            (op & CIGAR_OP_MASK) == BWA_CIGAR_OP_SOFT_CLIP
                || (op & CIGAR_OP_MASK) == BWA_CIGAR_OP_HARD_CLIP
        };
        // Operation lengths in bases, unpacked from the high bits. Meaningful only when the matching
        // `is_clip` test passes, which is why they are computed unconditionally but used guarded.
        let (first_len, last_len) = (
            (first_op >> CIGAR_LEN_SHIFT) as usize,
            (last_op >> CIGAR_LEN_SHIFT) as usize,
        );
        // On the reverse strand the cigar runs against the forward read, so the FIRST cigar op
        // trims the read's END and vice versa.
        if aln.is_rev {
            if is_clip(first_op) {
                qe -= first_len;
            }
            if is_clip(last_op) {
                qb += last_len;
            }
        } else {
            if is_clip(first_op) {
                qb += first_len;
            }
            if is_clip(last_op) {
                qe -= last_len;
            }
        }
    }

    // ---- Optional tags, in bwa's fixed order: NM MD AS XS RG SA XA <comment> ----
    // The optional-tag suffix of the record, built in bwa's fixed order and passed to the formatter
    // as one string. Every tag but the first is prefixed with its own `\t`; the formatter adds the
    // separator before the block. Empty is legal (an alignment with no cigar and a negative score).
    let mut tags = String::new();
    if !aln.cigar.is_empty() {
        tags.push_str(&format!("NM:i:{}\tMD:Z:{}", aln.nm, aln.md));
    }
    if aln.score >= 0 {
        tags.push_str(&format!("\tAS:i:{}", aln.score));
    }
    if aln.sub >= 0 {
        tags.push_str(&format!("\tXS:i:{}", aln.sub));
    }
    // `-R`: bwa emits RG:Z here, between XS and SA:Z.
    if let Some(id) = bwa_core::rg::rg_id() {
        tags.push_str("\tRG:Z:");
        tags.push_str(id);
    }
    // SA:Z lists this read's *other* emitted alignments, with their raw (unconverted) CIGARs.
    // bwa guards the whole block with `if (!(p->flag & 0x100))` and skips secondaries inside it,
    // so a secondary neither carries SA:Z nor appears in anyone else's.
    // Whether any OTHER non-secondary record exists for this read. Computed up front so the `SA:Z:`
    // prefix is only written when the group list that follows would be non-empty.
    let others_primary = alns
        .iter()
        .enumerate()
        .any(|(idx, other)| idx != which && !other.secondary);
    if !emit.secondary && others_primary {
        tags.push_str("\tSA:Z:");
        for (idx, other) in alns.iter().enumerate() {
            if idx == which || other.secondary {
                continue;
            }
            // One `rname,pos,strand,CIGAR,mapq,NM;` group per other alignment. `rid` indexes the
            // contig table; `pos + 1` because SAM's POS is 1-based while ours is 0-based.
            tags.push_str(&format!(
                "{},{},{},{},{},{};",
                bns.contigs[other.aln.rid as usize].name,
                other.aln.pos + 1,
                if other.aln.is_rev { '-' } else { '+' },
                cigar_string(&other.aln.cigar),
                other.mapq,
                other.aln.nm,
            ));
        }
    }
    // `pa:f` (`bwamem.cpp:1714`): how much better this primary-assembly hit scored than the ALT hit
    // that shadows it. Emitted for a non-secondary record whenever `alt_sc` was set, INDEPENDENTLY
    // of whether an `SA:Z` was printed just above, and always before `XA:Z`. `alt_sc` is 0 unless
    // the index has ALT contigs, so this is dead weight on an index without a `.alt` file.
    if !emit.secondary && aln.alt_sc > 0 {
        tags.push_str("\tpa:f:");
        tags.push_str(&bwa_mem::cigar::format_pa(aln.score, aln.alt_sc));
    }
    if let Some(xa_string) = &emit.xa {
        tags.push_str("\tXA:Z:");
        tags.push_str(xa_string);
    }
    // `-C`: bwa appends the FASTQ comment after everything else.
    if bwa_core::rg::copy_comment() {
        if let Some(comment) = rec.comment() {
            tags.push('\t');
            tags.push_str(comment);
        }
    }

    // ---- Remaining columns, then hand the finished pieces to the byte formatter ----
    // SAM column 6, already converted: `cigar_string_which` is what turns bwa's `H` into `S` for a
    // supplementary record under `-Y`, which is why it needs `which` and the softclip flag, unlike
    // the raw `cigar_string` used for `SA:Z` above.
    let cigar = cigar_string_which(&aln.cigar, which, emit.is_alt, softclip);
    // SAM column 3. `rid` is an index into the contig table, valid because a region that reached
    // here was placed on a real contig.
    let rname = &bns.contigs[aln.rid as usize].name;
    // bwa: "for secondary alignments, don't write SEQ and QUAL" -- the bases live on the primary.
    // On the reverse strand SEQ is reverse-COMPLEMENTED but QUAL is only REVERSED: a quality score
    // has no complement.
    // SAM columns 10 and 11, in the strand orientation the record is reported in. `qual` is `None`
    // for a FASTA input (no qualities) and for a secondary; the formatter writes `*` for it.
    let (seq, qual) = if emit.secondary {
        (b"*".to_vec(), None)
    } else if aln.is_rev {
        // Reversed but NOT complemented: a Phred score has no complement.
        let reversed_qual = rec.qual().map(|quals| {
            let mut sliced = quals[qb..qe].to_vec();
            sliced.reverse();
            sliced
        });
        (dna::revcomp_ascii(&rec.seq()[qb..qe]), reversed_qual)
    } else {
        (
            rec.seq()[qb..qe].to_vec(),
            rec.qual().map(|quals| quals[qb..qe].to_vec()),
        )
    };
    sam::write_mapped_se(
        buf,
        rec.name(),
        flag,
        rname,
        aln.pos + 1,
        emit.mapq,
        &cigar,
        &seq,
        qual.as_deref(),
        &tags,
    )
    .expect("write to Vec");
}

/// One batch of paired records plus the global id of its first pair, as the reader hands it over.
/// Named because the closure type that produces it is otherwise long enough to be unreadable.
type PeBatch = (Vec<(Record, Record)>, u64);

/// Sending half of the reader-to-pipeline channel for paired input.
type PeBatchSender = std::sync::mpsc::SyncSender<PeBatch>;

/// The paired-end reader's body, as a closure the caller can hand to [`spawn_reader`] BEFORE the
/// index is loaded.
///
/// Extracted from `run_pe` for exactly that reason: the reader used to be built inside the pipeline,
/// which meant the first batch's read, inflate and parse could not begin until the 10 GB index was
/// in memory. Nothing about what it produces changed.
///
/// # Parameters
/// * `reads1`, `reads2`: mate files, or `reads1` alone for interleaved input under `-p`. Owned, so
///   the closure borrows nothing and can be moved to a thread that outlives this call.
/// * `k_batch`: `-K` in input bases; fixes the batch boundaries and therefore the output.
fn pe_read_batches(
    reads1: std::path::PathBuf,
    reads2: Option<std::path::PathBuf>,
    k_batch: usize,
) -> impl FnOnce(PeBatchSender) -> anyhow::Result<()> + Send + 'static {
    move |tx: std::sync::mpsc::SyncSender<(Vec<(Record, Record)>, u64)>| -> anyhow::Result<()> {
        // Two files, or one interleaved file under `-p`.
        // Exactly one of these is `Some` for the life of the closure; the pair of `Option`s is
        // how the two reader types are held without a trait object. Both yield the same batch
        // type, so the rest of the loop is shared.
        let mut two: Option<PairedFastqReader> = None;
        let mut one: Option<InterleavedFastqReader> = None;
        match &reads2 {
            Some(r2) => two = Some(PairedFastqReader::from_paths(&reads1, r2)?),
            None => one = Some(InterleavedFastqReader::from_path(&reads1)?),
        }
        // Pairs (not reads) emitted in all previous batches, so the global 0-based id of this
        // batch's first pair. Invariant at the top of each turn: it equals the summed length of
        // every batch already sent. `mem_sam_pe` hashes it to break ties, so it must be global.
        let mut base_pair = 0u64;
        loop {
            // Up to `k_batch` input bases worth of PAIRS; empty only at end of input.
            let batch = match (two.as_mut(), one.as_mut()) {
                (Some(r), _) => r.next_batch(k_batch)?,
                (_, Some(r)) => r.next_batch(k_batch)?,
                _ => unreachable!("one reader is always set"),
            };
            if batch.is_empty() {
                break;
            }
            // Pair count, saved before the move so `base_pair` can advance after the send.
            let n = batch.len() as u64;
            if tx.send((batch, base_pair)).is_err() {
                break;
            }
            base_pair += n;
        }
        Ok(())
    }
}

/// Paired-end driver: per batch, align+dedup both ends of every pair, estimate insert sizes
/// (`mem_pestat`), then emit paired SAM (`mem_sam_pe`). The pair index is global across batches (for
/// the `hash` tie-break), matching bwa-mem2's `(n_processed>>1)+i`.
///
/// # Parameters
///
/// - `fm`, `bns`, `opt`: the shared immutable index halves and options, as everywhere.
/// - `reads1`: R1, or under `-p` the single interleaved file. Opened on the reader thread.
/// - `reads2`: R2, or `None` under `-p`. Presence here is what picks the reader type.
/// - `k_batch`: batch size in INPUT BASES, from `-K` or the default. Visible in the output, because
///   the insert-size model is re-estimated per batch.
/// - `backend`: seed-extension backend; byte-identical either way.
/// - `out`: the sink, already carrying the header, MOVED into the pipeline's writer thread.
/// - `pes0`: `-I`'s fixed insert-size distribution, or `None` to call `mem_pestat` per batch.
///
/// # Returns
///
/// `Ok(())` once every pair has been emitted and the sink finalized; errors as for [`run_pipeline`].
#[allow(clippy::too_many_arguments)]
fn run_pe(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    // The reader is already running when this is called: `run` starts it before loading the index
    // so the first batch's read, inflate and parse overlap the load. This is its receiving half.
    batch_rx: std::sync::mpsc::Receiver<PeBatch>,
    k_batch: usize,
    out: Output,
    // `-I`: user-supplied insert-size distribution. When present bwa copies it per batch and never
    // calls `mem_pestat` (`bwamem.cpp`: `if (pes0) memcpy(...) else mem_pestat(...)`).
    pes0: Option<[PeStat; 4]>,
    // `-v`, forwarded to `run_pipeline` for the batch-count line. `run_pe` has no `MemArgs` of its
    // own, so verbosity has to be threaded in from the caller.
    verbose: i32,
) -> anyhow::Result<()> {
    // Reader thread: open the mate files here and stream fixed-`-K` pair batches with the cumulative
    // pair id (global across batches for the `hash` tie-break, matching bwa-mem2's `(n_processed>>1)+i`).
    // Owned copies, so the closure can be moved onto the reader thread without borrowing the caller.
    // Seed -> chain -> BATCHED extension over both ends of every pair (interleaved c1,c2,...), then
    // per-read dedup. Regions are per-read independent, so this is byte-identical to the per-read
    // path; primary marking and pairing happen later, per bwa-mem2.
    // `batch` is one `-K` worth of read PAIRS in input order; `base_pair` is the global id of its
    // first pair. Returns the batch's complete SAM text.
    let process = |batch: Vec<(Record, Record)>, base_pair: u64| -> Vec<Vec<u8>> {
        // ---- Both mates of every pair, interleaved as c1,c2,c1,c2,... so the batched extension
        //      sees one flat list of reads ----
        // Length is `2 * batch.len()`; index `2i` is pair `i`'s mate 1 and `2i+1` its mate 2.
        //
        // Run on the pool, as bwa-mem2 does (it encodes inside `mem_kernel1_core`, under `kt_for`).
        // Indexed by the FLAT read index `i`, so `i >> 1` is the pair and `i & 1` the mate: an
        // indexed `par_iter` over a range preserves order exactly, and `dna::nt4` is a pure table
        // lookup, so each entry depends only on its own read. Byte-identical to the serial form.
        let all_codes: Vec<Vec<u8>> = stage_time::measure(Stage::Encode, || {
            stage_time::barrier::region(Stage::Encode, || {
                (0..2 * batch.len())
                    .into_par_iter()
                    .map(|i| {
                        stage_time::barrier::worker(|| {
                            let (rec1, rec2) = &batch[i >> 1];
                            let seq = if i & 1 == 0 { rec1.seq() } else { rec2.seq() };
                            seq.iter().map(|&base| dna::nt4(base)).collect()
                        })
                    })
                    .collect()
            })
        });
        // Both mates, so the bytes-per-base column counts the same bases the aligner processed.
        // One sum per batch, and only when `BWA4_STAGE_ALLOC` is armed.
        if crate::stage_alloc::enabled() {
            crate::stage_alloc::note_bases(all_codes.iter().map(|c| c.len() as u64).sum());
        }
        // Pre-dedup regions, in the same interleaved order as `all_codes`.
        let regs_all = stage_time::measure(Stage::Align, || batched_regs(fm, bns, opt, &all_codes));

        // ---- De-interleave: pair up each mate's owned codes + regions by sequential moves (no
        //      content copy), so the parallel prep can consume them instead of cloning
        //      `all_codes[2i]`/`regs_all[2i]` per pair ----
        // Owning cursors over the two interleaved lists. Invariant: both have yielded the same
        // number of items at every point, so the four `next()` calls below always take mate 1's
        // codes, mate 2's codes, mate 1's regions and mate 2's regions of the SAME pair. Each
        // `unwrap` is sound because both lists hold exactly `2 * batch.len()` items.
        // Re-paired, one entry per input pair, in pair order: ((codes1, codes2), (regs1, regs2)).
        #[allow(clippy::type_complexity)]
        let paired: Vec<((Vec<u8>, Vec<u8>), (Vec<MemAlnReg>, Vec<MemAlnReg>))> =
            stage_time::measure(Stage::Deinterleave, || {
                let mut code_iter = all_codes.into_iter();
                let mut regs_iter = regs_all.into_iter();
                (0..batch.len())
                    .map(|_| {
                        let codes1 = code_iter.next().unwrap();
                        let codes2 = code_iter.next().unwrap();
                        let regs1 = regs_iter.next().unwrap();
                        let regs2 = regs_iter.next().unwrap();
                        ((codes1, codes2), (regs1, regs2))
                    })
                    .collect()
            });

        // ---- Per-mate dedup. Regions are per-read independent, so this is byte-identical to the
        //      single-end path; pairing decisions come later ----
        // Everything the pairing and output stages need, one entry per pair in pair order. Mutable
        // because mate rescue writes new regions into it and `mem_sam_pe` then rewrites them again.
        let mut prepared: Vec<PrepPair> = stage_time::measure(Stage::DedupPrep, || {
            stage_time::barrier::region(Stage::DedupPrep, || {
                // `into_par_iter`, not `par_iter`: `batch` is dead after this stage, so the six
                // per-pair fields below are MOVED out of the records instead of cloned. At 1M pairs
                // that is six million allocations and about 376 MB of copying removed from a
                // parallel stage. Indexed either way, so the order is unchanged and so is the
                // output.
                batch
                    .into_par_iter()
                    .zip(paired.into_par_iter())
                    .map(
                        |((rec1, rec2), ((codes1, codes2), (regs_pre1, regs_pre2)))| {
                            stage_time::barrier::worker(|| {
                                let mut regs1 = mem_sort_dedup_patch(fm, opt, &codes1, regs_pre1);
                                let mut regs2 = mem_sort_dedup_patch(fm, opt, &codes2, regs_pre2);
                                // `bwamem.cpp:1161`, applied to both mates of the pair.
                                bwa_mem::stamp_is_alt(bns, &mut regs1);
                                bwa_mem::stamp_is_alt(bns, &mut regs2);
                                PrepPair {
                                    codes1,
                                    codes2,
                                    regs1,
                                    regs2,
                                    rec1,
                                    rec2,
                                }
                            })
                        },
                    )
                    .collect()
            })
        });

        // ---- Insert-size distribution, estimated once over the WHOLE batch. This is why batch
        //      boundaries (`-K`) are visible in paired-end output ----
        // Borrowed view of every mate's regions, re-interleaved, as `mem_pestat` expects: it reads
        // consecutive pairs of entries as the two ends of one pair.
        // Charged to `pestat` rather than to a row of its own: it exists only to feed `mem_pestat`
        // and shares its fate under any fix.
        let regs_ref: Vec<&[MemAlnReg]> = stage_time::measure(Stage::Pestat, || {
            prepared
                .iter()
                .flat_map(|pair| [pair.regs1.as_slice(), pair.regs2.as_slice()])
                .collect()
        });
        // The insert-size model in force for THIS batch: the four orientations' mean, std and the
        // properly-paired bounds, in bases. Fixed by `-I`, or inferred from this batch's own
        // unambiguous pairs. Read by both mate rescue and `mem_sam_pe`.
        let pes = match pes0 {
            Some(fixed) => fixed,
            None => stage_time::measure(Stage::Pestat, || mem_pestat(opt, bns.l_pac, &regs_ref)),
        };

        // Mate rescue, batched across the whole pair batch so the per-anchor insert-window SW fills
        // the SIMD lanes. Byte-identical to the per-pair rescue in `mem_sam_pe` (which is then told to
        // skip it). `BWA4_SCALAR_RESCUE` keeps the per-pair path for A/B verification.
        // `BWA4_SCALAR_RESCUE`: set to any value (presence only, content ignored) to skip the
        // batched pass here and let `mem_sam_pe` do its original per-pair rescue instead. Default
        // off. An A/B verification switch: both paths are byte-identical, only the speed differs.
        let scalar_rescue = std::env::var_os("BWA4_SCALAR_RESCUE").is_some();
        // `BWA4_NO_RESCUE=1` skips mate rescue entirely, the analogue of bwa-mem2's `-S`. It is a
        // MEASUREMENT gate, not a lever: it changes the output by design, exactly as `-S` does. It
        // exists so our rescue cost can be measured DIRECTLY and compared like-for-like against
        // `bwa-mem2 -S`, instead of decomposed as `PE - 2 x SE` (which assumes a read costs the same
        // to seed and extend in SE as in PE -- unverified).
        // `-S` (MEM_F_NO_RESCUE) is the user-facing form of the same gate.
        // Presence-only, like the gate above; the documented spelling is `BWA4_NO_RESCUE=1` but any
        // value works. Default off. True here means NO rescue happens anywhere, batched or per-pair.
        let no_rescue =
            std::env::var_os("BWA4_NO_RESCUE").is_some() || opt.flag & flags::NO_RESCUE != 0;
        if !scalar_rescue && !no_rescue {
            stage_time::measure(Stage::Rescue, || {
                // One rescue job per pair: each mate's codes plus a MUTABLE borrow of its region list,
                // which the rescue appends newly found alignments to. Borrowing `prepared` mutably is
                // why this vector exists at all rather than the rescue reading `prepared` directly.
                let mut rescue_jobs: Vec<PairRescueData> = prepared
                    .iter_mut()
                    .map(|pair| PairRescueData {
                        seq0: pair.codes1.as_slice(),
                        seq1: pair.codes2.as_slice(),
                        a0: &mut pair.regs1,
                        a1: &mut pair.regs2,
                    })
                    .collect();
                // Each pair's rescue is independent, so run chunks in parallel; a chunk of a few hundred
                // pairs still has enough rescue jobs to fill the SIMD lanes. Keeps -t8 scaling (the rescue
                // is otherwise a serial section) while byte-identical to the per-pair path.
                // Read before the mutable borrow below; `par_chunks_mut` would otherwise hold it.
                let chunk_pairs = rescue_pairs_per_chunk(rescue_jobs.len());
                stage_time::barrier::region(Stage::Rescue, || {
                    rescue_jobs.par_chunks_mut(chunk_pairs).for_each(|chunk| {
                        stage_time::barrier::worker(|| batch_mate_rescue(fm, bns, opt, &pes, chunk))
                    });
                });
            });
        }

        // Emit paired SAM in parallel (each pair owns its regions; global pair id fixes hashes).
        // One entry per pair, in pair order: that pair's SAM records (normally two, more when either
        // mate is split). rayon's indexed parallel iterator preserves the order.
        let bufs: Vec<Vec<u8>> = stage_time::measure(Stage::SamEmit, || {
            stage_time::barrier::region(Stage::SamEmit, || {
                prepared
                    .par_iter_mut()
                    .enumerate()
                    .map(|(pair_in_batch, pair)| {
                        stage_time::barrier::worker(|| {
                            // The two mates' per-record inputs, as [mate1, mate2] arrays because that is the
                            // shape `mem_sam_pe` takes (it indexes them 0/1 throughout).
                            let names = [pair.rec1.name(), pair.rec2.name()];
                            let seqs = [pair.codes1.as_slice(), pair.codes2.as_slice()];
                            let quals = [pair.rec1.qual(), pair.rec2.qual()];
                            let comments = [pair.rec1.comment(), pair.rec2.comment()];
                            // This pair's SAM text.
                            let mut buf = Vec::new();
                            mem_sam_pe(
                                fm,
                                bns,
                                opt,
                                &pes,
                                base_pair + pair_in_batch as u64,
                                &names,
                                &seqs,
                                &quals,
                                &comments,
                                &mut pair.regs1,
                                &mut pair.regs2,
                                // rescue_done: true when nothing further should rescue -- either the batched
                                // pass already did it, or BWA4_NO_RESCUE suppressed it outright.
                                !scalar_rescue || no_rescue,
                                &mut buf,
                            )
                            .expect("write to Vec");
                            buf
                        })
                    })
                    .collect()
            })
        });
        // Handed to the writer as pieces, in pair order (the indexed parallel map above preserved
        // it). No concatenation: see `run_pipeline`'s docs for why that buffer is not built.
        bufs
    };

    // The reader is joined by `run`, which owns its handle: it started the thread before the index
    // load, so it is also the place that observes a reader error.
    run_pipeline(out, batch_rx, process, verbose, k_batch)
}

/// Read pairs handed to one parallel mate-rescue task, derived from the batch rather than fixed.
///
/// This was a hard-coded 512 and that was measurably too small. Sweeping it on 500k pairs against
/// GRCh38 at `-t12`, two interleaved passes each, wall clock:
///
/// | pairs/chunk | 64 | 256 | 512 | 2048 | 4096 | 8192 | 16384 |
/// |---|---|---|---|---|---|---|---|
/// | seconds | 7.32 | 6.51 | 6.08 | **5.61** | 5.75 | 6.03 | 6.73 |
///
/// The curve is not about cache residency, which is what a constant would have been tuned for. It
/// is about how many chunks exist per worker. A `-K 10000000` batch holds roughly 33k pairs, so 512
/// makes ~65 chunks for 12 workers (scheduling overhead and lanes left underfilled) while 8192
/// makes 4, which leaves eight workers with nothing. Both ends of the sweep are that same effect.
///
/// So the figure has to follow the batch and the pool, not sit still: aim for a couple of chunks
/// per worker, which keeps work-stealing able to correct a straggler without paying for 65 task
/// boundaries. `RESCUE_MIN_PAIRS_PER_CHUNK` keeps a tiny final batch from being split into
/// per-worker slivers that cannot fill the SIMD lanes.
///
/// Purely a scheduling figure: the rescue result is independent of how the pairs are chunked,
/// verified byte-identical at 512 and 2048 on 200,003 records.
///
/// ## Why the constant is now overridable
///
/// The sweep above was run at `-t12` on a 33k-pair batch and never repeated. `BWA4_STAGE_TIME`
/// says it matters: on GRCh38, 500k pairs, `-K` 10M, PE, the two big stages scale very differently
/// from `-t1` to `-t12`:
///
/// | stage | `-t1` | `-t12` | speed-up | efficiency |
/// |---|---|---|---|---|
/// | align | 28.687s | 3.110s | 9.22x | 77% |
/// | **rescue** | 2.645s | **0.495s** | **5.35x** | **45%** |
///
/// Two chunks per worker is 24 tasks for 12 workers over a cost distribution whose mean is 207k DP
/// cells per job with a heavy tail, which is a straggler trap: one slow chunk holds the barrier and
/// there is nothing left to steal. That matters far more than the 10% share above suggests, because
/// these are wgsim reads, where `docs/perf-levers.md` measures mate rescue as nearly asleep; on real
/// GIAB it is 47-64% of paired-end compute.
///
/// ## Re-swept on REAL data (2026-07-29). Result: null. The formula above stands.
///
/// The sweep above used `work/r1_500k.fq`, i.e. wgsim, where `BWA4_STAGE_TIME` puts mate rescue at
/// **10%** of the wall. On real GIAB HG002 it is **59.5%** (`align` 34.1%), so the wgsim sweep tuned
/// the dominant stage of paired-end using a workload in which that stage is asleep. Re-swept on 1M
/// real GIAB pairs against GRCh38, and the conclusion is that this constant should NOT change:
///
/// A/B of the two candidate FORMULAS in the configuration that matters (`-t16`, DEFAULT `-K`, so
/// 540k pairs per batch: this formula yields 16892 pairs/chunk, a size-targeting one yields 768),
/// interleaved within each rep, `rescue` stage / total wall in seconds:
///
/// | rep | this formula | size-targeting |
/// |---|---|---|
/// | 1 | 15.622 / 27.566 | 14.480 / 26.167 |
/// | 2 | 16.184 / 27.962 | 15.843 / 28.759 |
/// | 3 | 19.241 / 32.908 | 16.611 / 29.033 |
///
/// Best-of-3 favours the alternative by 5.1% of wall; the MEDIAN favours this one by 2.8%. Drift of
/// one arm against ITSELF reaches 19% (27.566 -> 32.908), i.e. larger than the gap between arms. No
/// measurable difference, so the previously-tuned code is kept.
///
/// ### Two traps, recorded because both nearly produced a wrong commit
///
/// **Forcing a chunk SIZE on a fixed batch confounds size with count.** Sweeping sizes via the
/// override at `-K` 10M (33,784 pairs per batch) looked catastrophic at the top end (16384
/// pairs/chunk: rescue 68.3s vs 15.7s at 512, `-t16`). That is not a chunk-size cost: 16384 on a
/// 33,784-pair batch is **3 chunks for 16 workers**, so 13 workers idle. This formula can never
/// produce that, since it fixes the count at `2 * workers`. At the default `-K` the same 16892 costs
/// 15.6s, not 68s. Sweep the FORMULAS, not forced sizes, or fix the chunk count while varying size.
///
/// **The left branch is real and is the reason for the floor.** Below ~256 pairs the batched kernel
/// runs out of jobs to fill its lanes, and that IS a size effect: `-t12` rescue 36.3s at 64
/// pairs/chunk, 25.9s at 128, 21.6s at 256, then flat to 768. `RESCUE_MIN_PAIRS_PER_CHUNK` is what
/// keeps a small final batch out of that hole.
///
/// `BWA4_RESCUE_PAIRS_PER_CHUNK=<n>` overrides the result outright, so the figure stays sweepable on
/// ONE binary, the same arrangement as `BWA4_LOCKSTEP_N`. It bypasses the floor too, which is what
/// makes the sub-256 measurements above possible. It cannot change output for the reason already
/// given.
fn rescue_pairs_per_chunk(n_pairs: usize) -> usize {
    /// Two chunks per worker: enough for work-stealing to rebalance, few enough to keep the SIMD
    /// batches wide.
    const CHUNKS_PER_WORKER: usize = 2;
    /// Floor, so a small batch is not shredded into slivers narrower than the kernel's lanes.
    const RESCUE_MIN_PAIRS_PER_CHUNK: usize = 256;
    if let Some(forced) = rescue_pairs_override() {
        return forced;
    }
    let workers = rayon::current_num_threads().max(1);
    n_pairs
        .div_ceil(workers * CHUNKS_PER_WORKER)
        .max(RESCUE_MIN_PAIRS_PER_CHUNK)
}

/// `BWA4_RESCUE_PAIRS_PER_CHUNK`, the measurement override for [`rescue_pairs_per_chunk`]. Read once
/// and cached; 0 or an unparseable value means "not set" rather than an error, since this is a knob.
fn rescue_pairs_override() -> Option<usize> {
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("BWA4_RESCUE_PAIRS_PER_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
    })
}

/// One read pair prepared for the pairing/output stage: nt4 codes, dedup'd regions, names, quals.
struct PrepPair {
    /// nt4-coded bases of mate 1 and mate 2 (A=0, C=1, G=2, T=3, else 4).
    codes1: Vec<u8>,
    codes2: Vec<u8>,
    /// Each mate's deduplicated candidate alignments, still unpaired at this point. Mutated twice
    /// more: mate rescue may append to them, then `mem_sam_pe` marks primaries and pairs them.
    regs1: Vec<MemAlnReg>,
    regs2: Vec<MemAlnReg>,
    /// The two input records, MOVED out of the batch rather than copied field by field.
    ///
    /// This used to be six owned fields (`name1/2`, `qual1/2`, `comment1/2`) cloned out of the
    /// records, which is six allocations per pair. Since `Record` became one buffer plus offsets
    /// the whole record is 32 bytes and moving it is strictly cheaper than copying any one of its
    /// parts. `seq` rides along unused here (the stage works on `codes`), which costs nothing: the
    /// bytes were going to be freed at the end of the batch either way.
    ///
    /// `name()` is the QNAME: bwa emits the SAME name on both records of a pair, so any `/1`, `/2`
    /// suffix has already been stripped by the reader. `qual()` is the per-base ASCII quality, or
    /// `None` for FASTA input (SAM column 11 becomes `*`), always as long as the corresponding
    /// `codes`. `comment()` is the `-C` trailer.
    rec1: Record,
    rec2: Record,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The queue depths are a memory decision that is invisible in the output, so nothing else
    /// would catch a regression here. What is pinned is that the default is double buffering and
    /// that the override is honoured only when it parses to two usable depths.
    ///
    /// The env override itself is not exercised: `set_var` is process-global and the harness runs
    /// tests in threads, so setting it here would leak into whatever else is running.
    #[test]
    fn queue_depth_defaults_to_double_buffering() {
        assert_eq!(queue_depths(), (BATCH_READAHEAD, SAM_WRITEBEHIND));
        assert_eq!((BATCH_READAHEAD, SAM_WRITEBEHIND), (1, 1));
    }

    /// Two SAM records against a one-contig header, as the aligner would have formatted them.
    const HEADER: &str = "@SQ\tSN:chr1\tLN:100\n@PG\tID:bwa-mem4\tPN:bwa-mem4\tVN:0\tCL:test\n";
    const RECORDS: &str = "r1\t0\tchr1\t1\t60\t4M\t*\t0\t0\tACGT\tIIII\tNM:i:0\n\
                           r2\t16\tchr1\t7\t60\t4M\t*\t0\t0\tTTTT\tIIII\tNM:i:1\n";

    /// The BAM sink must reassemble records that a `write` boundary splits mid-line. The pipeline
    /// happens to hand it whole records today, so this is the only thing that pins the `pending`
    /// carry-over: feed the same bytes one byte at a time and the BAM must be the same file as
    /// feeding them in one call.
    #[test]
    fn bam_transcoder_reassembles_split_lines() {
        let dir = std::env::temp_dir();
        let whole = dir.join(format!("bwa3_bam_whole_{}.bam", std::process::id()));
        let split = dir.join(format!("bwa3_bam_split_{}.bam", std::process::id()));
        let text = format!("{HEADER}{RECORDS}");

        let mut out = Output::open(Some(&whole), 1, None).unwrap();
        out.write_all(text.as_bytes()).unwrap();
        out.finish().unwrap();

        let mut out = Output::open(Some(&split), 1, None).unwrap();
        for byte in text.as_bytes() {
            out.write_all(&[*byte]).unwrap();
        }
        out.finish().unwrap();

        let a = std::fs::read(&whole).unwrap();
        let b = std::fs::read(&split).unwrap();
        let _ = std::fs::remove_file(&whole);
        let _ = std::fs::remove_file(&split);
        assert_eq!(a, b, "byte-split writes must produce the same BAM");
        // Non-trivial output: the BGZF magic plus more than an empty file's EOF block.
        assert!(a.starts_with(&[0x1f, 0x8b]) && a.len() > 100);
    }

    /// A run that aligns nothing still owes the caller a valid, header-only BAM: the file is opened
    /// lazily on the first record, so `finish` has to force it.
    #[test]
    fn bam_header_only_still_produces_a_file() {
        let path = std::env::temp_dir().join(format!("bwa3_bam_hdr_{}.bam", std::process::id()));
        let mut out = Output::open(Some(&path), 1, None).unwrap();
        out.write_all(HEADER.as_bytes()).unwrap();
        out.finish().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        // BGZF magic, and long enough to hold the header block plus the 28-byte EOF block that
        // makes samtools consider the file complete rather than truncated.
        assert!(bytes.starts_with(&[0x1f, 0x8b]), "not a BGZF file");
        assert!(bytes.len() > 28, "no header block written");
    }

    /// CRAM without `--reference` must fail at option-parsing time, not halfway through a run.
    #[test]
    fn cram_without_reference_is_rejected() {
        let path = std::env::temp_dir().join("bwa3_never_created.cram");
        // `Output` is not `Debug`, so the error is pulled out by hand rather than via `unwrap_err`.
        let err = match Output::open(Some(&path), 1, None) {
            Ok(_) => panic!("CRAM without --reference must not open"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("--reference"), "unexpected message: {err}");
        assert!(!path.exists(), "the file must not have been created");
    }

    /// Parse a `mem` command line into [`MemArgs`], for the option-semantics tests below.
    ///
    /// `MemArgs` derives `Args`, not `Parser`, because it is a subcommand's body; wrapping it in a
    /// throwaway `Parser` is how clap exposes it to a test. The index prefix and read file are
    /// positional and required, so every call supplies them; neither is opened, because `build_opt`
    /// only reads the option fields.
    fn args_from(extra: &[&str]) -> MemArgs {
        use clap::Parser;
        #[derive(Parser)]
        struct Wrapper {
            #[command(flatten)]
            mem: MemArgs,
        }
        let mut argv: Vec<&str> = vec!["mem"];
        argv.extend_from_slice(extra);
        argv.extend_from_slice(&["idx.fa", "reads.fq"]);
        Wrapper::parse_from(argv).mem
    }

    /// `-x intractg` sets exactly the three groups bwa sets, and nothing else moves.
    ///
    /// Values from `fastmap.cpp:822-828`: gap open 16, mismatch 9, clip penalties 5. This is the one
    /// preset that still runs through bwa's own path here, so `scripts/opt_parity.sh` checks it
    /// end-to-end against the oracle too; this test pins the numbers so a failure says which field
    /// moved rather than just "the SAM differs".
    #[test]
    fn presets_match_bwa_intractg() {
        let opt = build_opt(&args_from(&["-x", "intractg"])).unwrap();
        assert_eq!((opt.o_del, opt.o_ins), (16, 16), "gap open");
        assert_eq!(opt.b, 9, "mismatch");
        assert_eq!((opt.pen_clip5, opt.pen_clip3), (5, 5), "clip penalty");
        // Untouched by this preset, so still at bwa's defaults.
        assert_eq!((opt.e_del, opt.e_ins), (1, 1), "gap extend");
        assert_eq!(opt.min_seed_len, 19, "min seed len");
    }

    /// The three long-read presets set what bwa sets, `ont2d` differing from `pacbio` in exactly two
    /// fields, and `pbref` being an alias of `pacbio`.
    ///
    /// These modes are routed to rammap by `run`, so nothing else in this binary exercises them.
    /// That is precisely why they are tested here: an unreachable branch that is silently wrong
    /// becomes a bug the moment the routing policy changes. Values from `fastmap.cpp:829-855`.
    #[test]
    fn presets_match_bwa_long_read() {
        for mode in ["pacbio", "pbref", "ont2d"] {
            let opt = build_opt(&args_from(&["-x", mode])).unwrap();
            let ont = mode == "ont2d";
            assert_eq!((opt.o_del, opt.o_ins), (1, 1), "{mode}: gap open");
            assert_eq!((opt.e_del, opt.e_ins), (1, 1), "{mode}: gap extend");
            assert_eq!(opt.b, 1, "{mode}: mismatch");
            assert_eq!(opt.split_factor, 10.0, "{mode}: split factor");
            assert_eq!(
                (opt.pen_clip5, opt.pen_clip3),
                (0, 0),
                "{mode}: clip penalty"
            );
            // The only two fields where ont2d and pacbio disagree.
            assert_eq!(
                opt.min_chain_weight,
                if ont { 20 } else { 40 },
                "{mode}: min chain weight"
            );
            assert_eq!(
                opt.min_seed_len,
                if ont { 14 } else { 17 },
                "{mode}: min seed len"
            );
        }
    }

    /// A preset leaves an explicitly-given option alone, and suppresses the `-A` rescaling.
    ///
    /// Both halves come from one line of C: `if (mode) { ... } else update_a(opt, &opt0);`. Written
    /// as two independent `if`s, `-x pacbio -A 2` would rescale the preset's values and diverge; the
    /// mismatch penalty is the visible symptom, 1 rather than 2.
    #[test]
    fn preset_defers_to_explicit_options_and_suppresses_rescaling() {
        // Explicit -B wins over the preset's mismatch penalty.
        let opt = build_opt(&args_from(&["-x", "intractg", "-B", "3"])).unwrap();
        assert_eq!(opt.b, 3, "explicit -B must survive the preset");
        assert_eq!(
            (opt.o_del, opt.o_ins),
            (16, 16),
            "the rest of the preset still applies"
        );

        // Explicit -k wins over ont2d's 14.
        let opt = build_opt(&args_from(&["-x", "ont2d", "-k", "25"])).unwrap();
        assert_eq!(opt.min_seed_len, 25, "explicit -k must survive the preset");

        // -A with a preset: the preset's values stand unscaled.
        let opt = build_opt(&args_from(&["-x", "pacbio", "-A", "2"])).unwrap();
        assert_eq!(opt.a, 2, "-A itself still applies");
        assert_eq!(opt.b, 1, "the preset's mismatch must NOT be rescaled by -A");

        // Without a preset, the same -A does rescale: this is the branch the preset suppresses.
        let scaled = build_opt(&args_from(&["-A", "2"])).unwrap();
        assert_eq!(scaled.b, 8, "-A 2 rescales the default mismatch of 4");
    }

    /// An unknown `-x` is rejected, as bwa's `[E::main_mem] unknown read type` is.
    #[test]
    fn unknown_read_type_is_rejected() {
        let err = build_opt(&args_from(&["-x", "nanopore"]))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("unknown read type"),
            "unexpected message: {err}"
        );
    }

    /// `-f` is bwa's bare alias of `-o`, and `-1` parses without affecting anything.
    ///
    /// Both are pure command-line compatibility: a script written for bwa-mem2 must not die here on
    /// an unknown argument. `-1` has no field to check beyond parsing, which is the whole claim.
    #[test]
    fn f_aliases_o_and_minus_one_parses() {
        let args = args_from(&["-f", "out.sam"]);
        assert_eq!(
            args.output.as_deref(),
            Some(std::path::Path::new("out.sam")),
            "-f must fill the same field as -o"
        );
        assert!(args_from(&["-1"]).no_mt_io, "-1 must parse");
    }
}
