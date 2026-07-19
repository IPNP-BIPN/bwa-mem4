//! Reference metadata parsing (`.ann` + `.amb`), mirroring bwa-mem2's `bntseq_t`.
//!
//! Formats (text), from `reference/bwa-mem2/src/bntseq.cpp`:
//! - `.ann`: line 1 `l_pac n_seqs seed`; then per contig two lines: `gi name anno`
//!   and `offset len n_ambs`. `anno` is literally `(null)` when absent.
//! - `.amb`: line 1 `l_pac n_seqs n_holes`; then per hole `offset len amb_char`.
//!
//! Both are written by `bns_dump` (`bntseq.cpp:73`) and read by `bns_restore_core`
//! (`bntseq.cpp:104`); see [`crate::build::write_ann`] / [`crate::build::write_amb`] for the exact
//! printf formats and the byte-identity traps. This module is the READ side.
//!
//! # What this structure is for
//!
//! The packed reference is one undelimited run of `l_pac` bases; nothing in it marks where one
//! chromosome ends and the next begins. `bntseq_t` is the only thing that knows, so every mapping
//! from a reference coordinate back to a (contig, position) pair, and every guard that stops an
//! alignment from crossing a contig boundary, goes through the functions below.
//!
//! # Coordinate spaces used here
//!
//! * FORWARD (or "1L") space: `[0, l_pac)`, positions in the concatenated forward reference. This
//!   is what `.ann` offsets are in, and what SAM POS is derived from.
//! * 2L space: `[0, 2*l_pac)`, the `forward ++ reverse_complement(forward)` array the FM index
//!   actually searches. `pos >= l_pac` means "reverse strand"; [`BntSeq::depos`] converts back.
//!   Seeds, chains and extension windows all live in 2L space, which is why the clamping helpers
//!   below take 2L coordinates and flip contig bounds onto the reverse half.
//!
//! # Glossary: names kept identical to the C
//!
//! | name | C origin | plain-language meaning |
//! |---|---|---|
//! | `l_pac` | `bns->l_pac` | Length of the FORWARD reference in bases: every contig of the FASTA concatenated head to tail, with no separator between them. |
//! | `pac` | `.pac` | The 2-bit packed forward reference. This module never reads it; it only describes where things are inside it. |
//! | `rid` | `bns_pos2rid` | Reference-sequence index, i.e. which contig a position falls in. Negative means "none". |
//! | `ann` / `anns` | `bntann1_t` | The per-contig table: name, annotation, start offset, length. |
//! | `amb` / `ambs` | `bntamb1_t` | The ambiguous-base ("hole") table: runs of N (or any non-ACGT letter) that the builder replaced with random bases. |
//! | `n_seqs` | `bns->n_seqs` | Number of contigs. |
//! | `depos` | `bns_depos` | "De-position": map a 2L-space coordinate back to a forward coordinate plus a strand flag. |
//!
//! # Reading order for this file
//!
//! 1. [`Contig`] and [`Amb`]: what the two tables hold.
//! 2. [`BntSeq::depos`]: the 2L to forward coordinate mapping every other method builds on.
//! 3. [`BntSeq::pos2rid`], then [`BntSeq::intv2rid`], then [`BntSeq::fetch_bounds`].
//! 4. [`parse_ann`] / [`parse_amb`]: the file readers, which are straightforward.

use bwa_core::{Error, Result};
use std::path::{Path, PathBuf};

/// One reference contig (from `.ann`), the C's `bntann1_t`.
#[derive(Debug, Clone)]
pub struct Contig {
    /// GenBank gi number. Always 0: bwa hard-codes `p->gi = 0` (`bntseq.cpp:262`) and nothing reads
    /// it back. Kept only so the field order matches the file.
    pub gi: i64,
    /// The contig name, i.e. the FASTA header up to the first space. This becomes SAM `@SQ SN:` and
    /// the RNAME column, so it must round-trip verbatim.
    pub name: String,
    /// The rest of the FASTA header line, or the literal `(null)` when there was none. Never empty.
    pub anno: String,
    /// Start of this contig in FORWARD space. Contigs tile `[0, l_pac)` in ascending order with no
    /// gaps, which is the invariant [`BntSeq::pos2rid`]'s binary search depends on.
    pub offset: i64,
    /// Length in bases. A SAM position is `pos_forward - offset` (plus 1 for SAM's 1-based POS).
    pub len: i32,
    /// Number of ambiguous-base RUNS inside this contig (not the number of ambiguous bases).
    pub n_ambs: i32,
}

/// An ambiguous-base run (from `.amb`), the C's `bntamb1_t`. Also called a "hole".
#[derive(Debug, Clone)]
pub struct Amb {
    /// Start in FORWARD space, absolute across the whole reference (not contig-relative).
    pub offset: i64,
    /// Run length in bases.
    pub len: i32,
    /// The original FASTA character (`N` usually, but any non-ACGT letter). The packed reference
    /// holds a RANDOM ACGT at these positions instead (see [`crate::build`]), so this file is the
    /// only surviving evidence that the bases were unknown.
    pub amb: char,
}

/// Reference metadata parsed from `<prefix>.ann` and `<prefix>.amb`.
#[derive(Debug, Clone)]
pub struct BntSeq {
    /// Total length of the FORWARD reference in bases, the sum of all contig lengths. The FM index
    /// searches `2 * l_pac` bases and its `ref_seq_len` header is `2 * l_pac + 1`.
    pub l_pac: i64,
    /// Number of contigs; equals `contigs.len()`.
    pub n_seqs: i32,
    /// The RNG seed recorded at build time, always 11. Informational only.
    pub seed: u32,
    /// One entry per contig, in reference order (ascending `offset`).
    pub contigs: Vec<Contig>,
    /// Number of ambiguous runs; equals `ambs.len()`.
    pub n_holes: i32,
    /// Ambiguous runs in ascending, non-overlapping `offset` order.
    pub ambs: Vec<Amb>,
}

impl BntSeq {
    /// Load `<prefix>.ann` and `<prefix>.amb`. `prefix` is the FASTA path passed to `index`; the
    /// index files are siblings named `<prefix>.ann`, `<prefix>.amb`, ...
    ///
    /// # Parameters
    ///
    /// * `prefix`: the index prefix, i.e. the path of the FASTA that was indexed, WITHOUT any
    ///   extension appended. Supplied by the caller from the command line (`bwa mem <prefix> ...`).
    ///   Both `<prefix>.ann` and `<prefix>.amb` must exist and must come from the same `index` run;
    ///   a mismatched pair is not detected here (see [`parse_amb`]).
    ///
    /// # Returns
    ///
    /// A fully populated `BntSeq` (both the contig table and the hole table). Errors are I/O errors
    /// from either file, or [`Error::IndexFormat`] when a field is missing or unparseable.
    pub fn load(prefix: &Path) -> Result<Self> {
        let ann = std::fs::read_to_string(sibling(prefix, "ann"))?;
        let amb = std::fs::read_to_string(sibling(prefix, "amb"))?;
        let mut bns = parse_ann(&ann)?;
        parse_amb(&amb, &mut bns)?;
        Ok(bns)
    }

    /// Map a 2L-space position to `(forward_pos, is_rev)`, mirroring `bns_depos`.
    ///
    /// `pos` must be in `[0, 2*l_pac)`. The reverse half is stored reversed AND complemented, so
    /// the mirror is `2*l_pac - 1 - pos`, not `pos - l_pac`: 2L position `l_pac` is the LAST
    /// forward base, and 2L position `2*l_pac - 1` is the first.
    ///
    /// Micro-example with `l_pac = 4` (forward ACGT, 2L array ACGT|ACGT):
    ///   `depos(0) = (0, false)`, `depos(4) = (3, true)`, `depos(7) = (0, true)`.
    ///
    /// # Parameters
    ///
    /// * `pos`: a position in 2L space, in bases, valid range `[0, 2*l_pac)`. Supplied by seeding,
    ///   chaining or extension, all of which work in 2L space because that is what the FM index
    ///   returns. Out-of-range input is not checked: a negative `pos` is returned unchanged as a
    ///   forward position, and a `pos >= 2*l_pac` yields a negative forward position.
    ///
    /// # Returns
    ///
    /// `(forward_pos, is_rev)`: `forward_pos` is in FORWARD space, `[0, l_pac)`, in bases, and is
    /// the one to feed to [`pos2rid`] or to turn into SAM POS. `is_rev` is true exactly when `pos`
    /// came from the reverse-complement half, i.e. the alignment is on the minus strand.
    #[inline]
    pub fn depos(&self, pos: i64) -> (i64, bool) {
        if pos >= self.l_pac {
            ((self.l_pac << 1) - 1 - pos, true)
        } else {
            (pos, false)
        }
    }

    /// Clamp a 2L-space reference window `[beg, end)` to the contig containing `mid`, mirroring the
    /// clamp inside `bns_fetch_seq`. Returns `(beg, end, rid)`.
    ///
    /// This is what stops an extension from running off the end of a contig into whatever sequence
    /// the packed reference happens to hold next: bwa resolves the seed's contig, then trims the
    /// window to that contig's span (flipped onto the reverse strand when the seed is reverse). It
    /// matters wherever a read reaches a contig edge, most visibly on the circular MT genome.
    ///
    /// THE KEY POINT, and an easy thing to get wrong when porting: clamping merely to
    /// `[0, l_pac<<1)` and to the forward/reverse midpoint is NOT enough. The contigs are
    /// concatenated with no separator, so a window clamped only to the global bounds happily
    /// includes bases belonging to the NEXT chromosome, and the extension aligns the read against
    /// a junction sequence that does not exist in the genome. The C is unambiguous here
    /// (`bntseq.cpp:463-472`): it computes `far_beg`/`far_end` from `bns->anns[*rid]`, the seed's
    /// OWN contig, and only then does `*beg = max(*beg, far_beg)` / `*end = min(*end, far_end)`.
    ///
    /// # Parameters
    ///
    /// All three are positions in 2L space, in bases, valid range `[0, 2*l_pac)`.
    /// * `beg`, `end`: the half-open window the caller wants, `beg < end`. Supplied by the extension
    ///   code as "seed position plus/minus enough room for the unaligned read tail", so it routinely
    ///   runs past a contig edge, which is the whole reason this function exists. The C additionally
    ///   swaps them if reversed and asserts `beg <= mid < end` (`bntseq.cpp:459-461`); this port
    ///   expects the caller to have ordered them already.
    /// * `mid`: a position KNOWN to lie inside the seed, used to pick the contig. It must be the
    ///   seed itself and not the window centre, because the window is what may be out of bounds.
    ///
    /// # Returns
    ///
    /// The clamped `(beg, end, rid)`: `beg`/`end` still in 2L space and now guaranteed inside the
    /// contig `rid` (on the correct strand half), and `rid` the contig index in `[0, n_seqs)`.
    /// A negative `rid` (position at or past `l_pac` in forward space) passes the window through
    /// unclamped, matching the C's behaviour of trusting `bns_pos2rid`; callers must treat
    /// `rid < 0` as "no valid contig".
    ///
    /// The reverse flip is the interval image of [`depos`]: contig span `[o, o+len)` on the forward
    /// strand becomes `[2*l_pac - (o+len), 2*l_pac - o)` on the reverse strand. Note it uses
    /// `2*l_pac - x`, without the `-1` that the single-position mapping has, because reflecting a
    /// half-open interval swaps which endpoint is inclusive and the two off-by-ones cancel.
    pub fn fetch_bounds(&self, beg: i64, end: i64, mid: i64) -> (i64, i64, i32) {
        // `mid_f` is the seed position in FORWARD space (`[0, l_pac)`), which is the only space
        // `pos2rid` and the `.ann` offsets are expressed in. `is_rev` records which half of the 2L
        // array `mid` came from, and decides whether the contig span has to be mirrored below.
        let (mid_f, is_rev) = self.depos(mid);
        // Contig index of the SEED (not of the window ends, which may lie in a neighbouring contig).
        let rid = self.pos2rid(mid_f);
        if rid < 0 {
            return (beg, end, rid);
        }
        let contig = &self.contigs[rid as usize];
        // The seed's contig span as a half-open interval, currently in FORWARD space:
        // `[offset, offset + len)`.
        let (mut far_beg, mut far_end) =
            (contig.offset, contig.offset + i64::from(contig.len));
        if is_rev {
            // Save the forward start before overwriting `far_beg`, because the mirrored END is
            // computed from it: reflection swaps the two endpoints.
            let fwd_beg = far_beg;
            far_beg = (self.l_pac << 1) - far_end;
            far_end = (self.l_pac << 1) - fwd_beg;
        }
        // `far_beg`/`far_end` are now the contig span in 2L space on the seed's own strand half.
        // Intersect the requested window with it.
        (beg.max(far_beg), end.min(far_end), rid)
    }

    /// Contig index containing forward position `pos_f`, or -1, mirroring `bns_pos2rid`.
    ///
    /// `pos_f` must be in FORWARD space (`[0, l_pac)`); pass `depos(p).0`, never a raw 2L position.
    ///
    /// The loop is a deliberately literal transcription of the C's hand-rolled binary search
    /// (`bntseq.cpp`, `bns_pos2rid`) rather than an idiomatic `partition_point`. Two quirks are
    /// preserved on purpose: `mid` is declared outside the loop and RETURNED after it, so the
    /// result is whatever the last probe was; and the search terminates early via `break` when
    /// `pos_f` falls between `offset[mid]` and `offset[mid+1]`, with a special case for the last
    /// contig (which has no successor to compare against). A `partition_point` rewrite would agree
    /// on well-formed input, but this shape is guaranteed to agree on malformed input too.
    ///
    /// INVARIANT: contig offsets are strictly ascending and start at 0. Violate it and this
    /// silently returns the wrong contig, which surfaces as reads mapped to the wrong chromosome
    /// rather than as a crash.
    ///
    /// # Parameters
    ///
    /// * `pos_f`: a FORWARD-space position, in bases, valid range `[0, l_pac)`. Supplied by callers
    ///   that have already run [`depos`]. Passing a 2L-space position instead is the classic bug:
    ///   anything at or past `l_pac` simply returns -1, and anything below it silently names the
    ///   wrong contig.
    ///
    /// # Returns
    ///
    /// The contig index (`rid`) in `[0, n_seqs)`, or -1 when `pos_f` is not inside the forward
    /// reference at all.
    pub fn pos2rid(&self, pos_f: i64) -> i32 {
        if pos_f >= self.l_pac {
            return -1;
        }
        // Bracket over CONTIG INDICES (not positions): the answer is known to lie in `[left, right)`
        // at the top of every iteration, because contig offsets ascend and `pos_f < l_pac`.
        // `mid` is deliberately declared out here: it survives the loop and IS the return value,
        // so its final content is "the contig index of the last probe", which the `break`s below
        // arrange to be the correct one.
        let mut left = 0i32;
        let mut mid = 0i32;
        let mut right = self.n_seqs;
        while left < right {
            mid = (left + right) >> 1;
            if pos_f >= self.contigs[mid as usize].offset {
                // At or past contig `mid`'s start. Two ways this is already the answer: `mid` is the
                // last contig (nothing after it to fall into), or `pos_f` stops short of the next
                // contig's start. Otherwise the answer is strictly to the right.
                if mid == self.n_seqs - 1 {
                    break;
                }
                if pos_f < self.contigs[(mid + 1) as usize].offset {
                    break;
                }
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        mid
    }

    /// Contig index for the interval `[rb, re)` in 2L space, or a negative code if it bridges a
    /// contig boundary (-1) or the forward/reverse midpoint (-2). Mirrors `bns_intv2rid`.
    ///
    /// Used as a validity filter: a seed or chain that spans two contigs, or that straddles the
    /// forward/reverse junction at `l_pac`, does not correspond to any real genomic locus and must
    /// be discarded rather than clamped. The two failure codes are distinct because the callers
    /// treat them differently, and callers must test `rid < 0`, not `rid == -1`.
    ///
    /// `re` is EXCLUSIVE, hence `re - 1` when resolving the end position: using `re` itself would
    /// report a spurious boundary crossing for an interval that ends exactly on a contig edge.
    /// The `rb < re` guard covers the degenerate empty interval, for which the C reuses `rid_b`.
    ///
    /// # Parameters
    ///
    /// * `rb`: interval start, 2L space, in bases, `[0, 2*l_pac)`, INCLUSIVE. Supplied by the
    ///   seeding/chaining code as a chain's leftmost reference coordinate.
    /// * `re`: interval end, 2L space, in bases, EXCLUSIVE, normally `> rb`.
    ///
    /// # Returns
    ///
    /// The single contig index in `[0, n_seqs)` that contains the whole interval; -1 if the two
    /// ends land in different contigs; -2 if the interval straddles `l_pac`, i.e. the
    /// forward/reverse junction.
    pub fn intv2rid(&self, rb: i64, re: i64) -> i32 {
        if rb < self.l_pac && re > self.l_pac {
            return -2;
        }
        // Contig index of the interval's first base, resolved in FORWARD space via `depos`.
        let rid_b = self.pos2rid(self.depos(rb).0);
        // Contig index of the interval's LAST base (`re - 1`, since `re` is exclusive).
        let rid_e = if rb < re {
            self.pos2rid(self.depos(re - 1).0)
        } else {
            rid_b
        };
        if rid_b == rid_e {
            rid_b
        } else {
            -1
        }
    }
}

/// Build the path `<prefix>.<ext>`, the naming scheme every bwa index file follows.
///
/// # Parameters
///
/// * `prefix`: the index prefix (the indexed FASTA's path), as given on the command line.
/// * `ext`: the extension WITHOUT its leading dot, e.g. `"ann"` or `"amb"`.
///
/// # Returns
///
/// `prefix` with `"." + ext` appended. It appends to the OS string rather than using
/// `Path::with_extension`, which would REPLACE an existing extension: `ref.fa` must become
/// `ref.fa.ann`, not `ref.ann`.
fn sibling(prefix: &Path, ext: &str) -> PathBuf {
    let mut s = prefix.as_os_str().to_owned();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

/// Wrap a description of a malformed index file into the crate's error type.
///
/// # Parameters
///
/// * `what`: human-readable description of which field or line failed, e.g. `".ann l_pac"`. It is
///   surfaced to the user verbatim, so it should name the file and the field.
fn fmt_err(what: &str) -> Error {
    Error::IndexFormat(what.to_string())
}

/// Parse `.ann`. Returns a `BntSeq` with the `.amb` half still empty; [`parse_amb`] fills it.
///
/// Mirrors the `.ann` block of `bns_restore_core` (`bntseq.cpp:104-160`), which `fscanf`s the same
/// fields in the same order. `n_holes`/`ambs` are left at zero here because the C also builds the
/// two halves separately.
///
/// # Parameters
///
/// * `text`: the entire `.ann` file as UTF-8 text, already read into memory by [`BntSeq::load`].
///   Expected shape is one header line then exactly `2 * n_seqs` further lines; a short file is
///   reported as [`Error::IndexFormat`] rather than truncating the contig table.
///
/// # Returns
///
/// A `BntSeq` whose `l_pac`, `n_seqs`, `seed` and `contigs` are populated and whose `n_holes` /
/// `ambs` are still zero/empty.
fn parse_ann(text: &str) -> Result<BntSeq> {
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| fmt_err(".ann: empty file"))?;
    let mut header_fields = header.split_whitespace();
    // Header line 1 of `.ann`: total FORWARD reference length in bases, contig count, build seed.
    let l_pac: i64 = next_parse(&mut header_fields, ".ann l_pac")?;
    let n_seqs: i32 = next_parse(&mut header_fields, ".ann n_seqs")?;
    let seed: u32 = next_parse(&mut header_fields, ".ann seed")?;

    // `n_seqs` is read from an untrusted file, so `max(0)` keeps a negative count from panicking in
    // the capacity computation; the loop below would then simply not execute.
    let mut contigs = Vec::with_capacity(n_seqs.max(0) as usize);
    for i in 0..n_seqs {
        let name_line = lines
            .next()
            .ok_or_else(|| fmt_err(&format!(".ann: missing name line for contig {i}")))?;
        // "<gi> <name> <anno...>": contig names contain no spaces, anno is the remainder.
        // `splitn(3, ' ')` (not `split_whitespace`) is required: the annotation is free text that
        // may contain spaces AND tabs, and must be captured verbatim as one field. Splitting on all
        // whitespace would truncate it at the first space and change the SAM header we emit.
        let mut name_fields = name_line.splitn(3, ' ');
        let gi: i64 = name_fields
            .next()
            .and_then(|x| x.parse().ok())
            .ok_or_else(|| fmt_err(".ann: bad gi"))?;
        let name = name_fields
            .next()
            .ok_or_else(|| fmt_err(".ann: bad name"))?
            .to_string();
        // DIVERGENCE (benign, and deliberate): the C NORMALIZES on read, turning the placeholder
        // back into an empty string (`if (q - str > 1 && strcmp(str, " (null)") != 0)
        // p->anno = strdup(str + 1); else p->anno = strdup("");`, `bntseq.cpp:143`). We keep the
        // literal `(null)`, which round-trips `.ann` verbatim if it is ever rewritten. Safe because
        // nothing downstream reads `anno`: the only consumer in the tree is `build::write_ann`,
        // and the SAM header uses `name`/`len` only.
        let anno = name_fields.next().unwrap_or("").to_string();

        let stat_line = lines
            .next()
            .ok_or_else(|| fmt_err(&format!(".ann: missing offset line for contig {i}")))?;
        let mut stat_fields = stat_line.split_whitespace();
        // Second line per contig: start in FORWARD space (bases, `[0, l_pac)`), length in bases,
        // and the count of ambiguous RUNS inside this contig. All three are plain integers here,
        // so `split_whitespace` is fine (unlike the name line above).
        let offset: i64 = next_parse(&mut stat_fields, ".ann offset")?;
        let len: i32 = next_parse(&mut stat_fields, ".ann len")?;
        let n_ambs: i32 = next_parse(&mut stat_fields, ".ann n_ambs")?;

        contigs.push(Contig {
            gi,
            name,
            anno,
            offset,
            len,
            n_ambs,
        });
    }

    Ok(BntSeq {
        l_pac,
        n_seqs,
        seed,
        contigs,
        n_holes: 0,
        ambs: Vec::new(),
    })
}

/// Parse `.amb` into an already-parsed `BntSeq`.
///
/// The header's `l_pac` and `n_seqs` duplicate `.ann`; only the third field, `n_holes`, is new
/// information. DIVERGENCE (benign): the C cross-checks the duplicates and aborts on mismatch
/// (`xassert(l_pac == bns->l_pac && n_seqs == bns->n_seqs, "inconsistent .ann and .amb files.")`,
/// `bntseq.cpp:159`), whereas this port parses them into `_`-prefixed bindings and drops them. That
/// only loosens an error check on a corrupt index; it cannot change output for a well-formed one.
///
/// # Parameters
///
/// * `text`: the entire `.amb` file as UTF-8 text. Expected shape is one header line then exactly
///   `n_holes` further lines.
/// * `bns`: the `BntSeq` produced by [`parse_ann`], mutated in place. Its `n_holes` and `ambs` are
///   overwritten/appended; nothing else is touched. Passing a `BntSeq` from a DIFFERENT index is
///   not detected (see the divergence note above).
///
/// # Returns
///
/// `Ok(())` on success; [`Error::IndexFormat`] on a missing or unparseable field. On error `bns`
/// may already be partially filled, which is acceptable because the caller discards it.
fn parse_amb(text: &str, bns: &mut BntSeq) -> Result<()> {
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| fmt_err(".amb: empty file"))?;
    let mut header_fields = header.split_whitespace();
    // The first two header fields duplicate `.ann` and are parsed only to advance past them; the
    // `_` prefix records that they are intentionally dropped (see the DIVERGENCE note above).
    let _l_pac: i64 = next_parse(&mut header_fields, ".amb l_pac")?;
    let _n_seqs: i32 = next_parse(&mut header_fields, ".amb n_seqs")?;
    // The only new information in this header: how many ambiguous RUNS follow, one per line.
    let n_holes: i32 = next_parse(&mut header_fields, ".amb n_holes")?;
    bns.n_holes = n_holes;
    for i in 0..n_holes {
        let hole_line = lines
            .next()
            .ok_or_else(|| fmt_err(&format!(".amb: missing hole line {i}")))?;
        let mut hole_fields = hole_line.split_whitespace();
        // Hole start in FORWARD space, absolute across the whole reference (NOT contig-relative),
        // and its length in bases.
        let offset: i64 = next_parse(&mut hole_fields, ".amb offset")?;
        let len: i32 = next_parse(&mut hole_fields, ".amb len")?;
        // The original FASTA letter for the run. Defaulting to 'N' on a missing/empty field is a
        // tolerance this port adds; 'N' is what essentially every real hole holds.
        let amb = hole_fields
            .next()
            .and_then(|x| x.chars().next())
            .unwrap_or('N');
        bns.ambs.push(Amb { offset, len, amb });
    }
    Ok(())
}

/// Pull the next whitespace-separated field off `it` and parse it as `T`, turning both "ran out of
/// fields" and "not a number" into a descriptive [`Error::IndexFormat`].
///
/// # Parameters
///
/// * `it`: the field iterator for the current line, advanced by exactly one field on success and
///   on a parse failure alike.
/// * `what`: label used in the error message, by convention `"<file> <field>"`, e.g. `".ann len"`.
///
/// # Returns
///
/// The parsed value, or an error naming `what`. `T` is inferred from the binding at the call site,
/// which is why every caller annotates its `let` with an explicit type.
fn next_parse<'a, T, I>(it: &mut I, what: &str) -> Result<T>
where
    T: std::str::FromStr,
    I: Iterator<Item = &'a str>,
{
    it.next()
        .ok_or_else(|| fmt_err(&format!("{what}: missing field")))?
        .parse::<T>()
        .map_err(|_| fmt_err(&format!("{what}: parse error")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_committed_tiny_index() {
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        let bns = BntSeq::load(Path::new(prefix)).unwrap();
        assert_eq!(bns.l_pac, 200_001);
        assert_eq!(bns.n_seqs, 1);
        assert_eq!(bns.seed, 11);
        assert_eq!(bns.n_holes, 0);
        assert_eq!(bns.contigs.len(), 1);
        assert_eq!(bns.contigs[0].name, "20:2000000-2200000");
        assert_eq!(bns.contigs[0].len, 200_001);
        assert_eq!(bns.contigs[0].offset, 0);
    }
}
