//! SAM concordance: compare two SAM files field-by-field and summarise agreement.
//!
//! This is the inner-loop oracle-diff used to gate every phase. It compares primary alignments
//! (skipping secondary/supplementary) keyed by QNAME + read-in-pair bit.
//!
//! # What this is and is not for
//!
//! The project's real gate is byte-identity: `cmp` the two SAM files. This tool is the DIAGNOSTIC
//! for when that fails. Byte-identity tells you only that something differs; this tells you which
//! reads, in which field, and with what values, which is what you can actually act on.
//!
//! Consequently it is deliberately lossy. It looks at five fields (FLAG, RNAME, POS, MAPQ, CIGAR)
//! and ignores tags, SEQ and QUAL entirely; it drops secondary and supplementary records; and it is
//! order-insensitive, since it keys into a `HashMap`. A run that reports 100% concordance here can
//! still fail the byte gate, for instance on a differing `NM:i:` tag or a different record order.
//! Never read a clean report as proof of parity.
//!
//! # Glossary
//!
//! | Term | Plain language |
//! |------|----------------|
//! | oracle | the trusted output, i.e. bwa-mem2's SAM; "ours" is the candidate under test |
//! | FLAG | SAM column 2, a bitfield; 0x100 = secondary, 0x800 = supplementary, 0x80 = second mate |
//! | primary | the one record per read that is neither secondary nor supplementary |
//! | secondary | an alternative placement of the same read, shadowed by a better one |
//! | supplementary | part of a read that aligned somewhere else (a split/chimeric alignment) |
//! | RNAME / POS | which contig, and the 1-based leftmost base on it |
//! | MAPQ | confidence the position is right, Phred-scaled: 60 is high, 0 means "maps anywhere" |
//! | CIGAR | the run-length alignment ops, e.g. `100M` or `30S70M` |
//!
//! Reading order: [`Rec`] (what is compared), [`parse_primary`] (SAM in, keyed map out), then
//! [`compare`] (two maps in, [`Report`] out). The binary `sam_diff` is a thin argv wrapper.

use std::collections::HashMap;
use std::path::Path;

use serde::Serialize;

/// FLAG bit 0x100: this is a secondary alignment, an alternative placement of a read that is
/// already reported elsewhere. Skipped, because only primaries are gated.
const FLAG_SECONDARY: u32 = 0x100;

/// FLAG bit 0x800: this is a supplementary alignment, another piece of a split read. Skipped for
/// the same reason.
const FLAG_SUPPLEMENTARY: u32 = 0x800;

/// FLAG bit 0x80: "last segment in the template", i.e. this record is mate 2 of its pair. Used to
/// split the shared QNAME into two distinct keys.
const FLAG_LAST_SEGMENT: u32 = 0x80;

/// A minimal primary-alignment record (the fields we gate on).
///
/// SAM columns 2, 3, 4, 5 and 6. `pos` is 1-based as it appears in the file (no conversion), and
/// `rname`/`cigar` keep the literal `*` when absent, so a missing value compares equal to a missing
/// value rather than to position 0 on some contig.
#[derive(Debug, Clone, PartialEq)]
pub struct Rec {
    /// SAM column 2 (FLAG): the record's bitfield, exactly as parsed. Range 0..=4095 in practice.
    /// Set once in [`parse_primary`], and by construction here bits 0x100 (secondary) and 0x800
    /// (supplementary) are always clear, since such records are dropped before a `Rec` is built.
    /// Read by [`compare`] for the `flag_match` verdict.
    pub flag: u32,
    /// SAM column 3 (RNAME): the reference contig name, e.g. `chr1`, or the literal `*` for an
    /// unmapped record. Kept as the raw string with no normalisation, so `*` compares equal only
    /// to `*`. Compared jointly with `pos`.
    pub rname: String,
    /// SAM column 4 (POS): the 1-based leftmost mapped base on `rname`, as written in the file with
    /// no coordinate conversion. 0 means unplaced (and is also what an unparseable field yields).
    /// Meaningful only together with `rname`: the same integer on a different contig is a different
    /// location, which is why [`compare`] treats the pair as one verdict.
    pub pos: i64,
    /// SAM column 5 (MAPQ): Phred-scaled mapping confidence, 0..=255 per the spec and 0..=60 for
    /// BWA-MEM. 0 means the read maps equally well elsewhere; 255 means unavailable.
    pub mapq: u32,
    /// SAM column 6 (CIGAR): the run-length alignment operations, e.g. `100M` or `30S70M`, or the
    /// literal `*` when absent. Compared as an exact string, so `100M` and `50M50M` differ even
    /// though they describe the same alignment.
    pub cigar: String,
}

/// A single field-level divergence example.
///
/// One of these is emitted per divergent record (never more), for the first `max_examples`
/// divergent records only. All four members are strings so the struct can carry any field's value
/// uniformly and serialise straight to JSON.
#[derive(Debug, Serialize)]
pub struct Divergence {
    /// The map key that diverged, `"<qname>/<1|2>"`: SAM column 1 (QNAME) plus the mate number
    /// derived from FLAG bit 0x80. This is what you grep for in both SAM files to see the two
    /// records side by side.
    pub key: String,
    /// Which field this example is attributed to: one of the literals `"RNAME/POS"`, `"CIGAR"`,
    /// `"MAPQ"`, `"FLAG"`. It is the ROOT-MOST differing field in that priority order, not the only
    /// one differing, so a record may also mismatch on fields listed after this one.
    pub field: String,
    /// The oracle's (bwa-mem2's) rendering of `field`. For `"RNAME/POS"` this is the composite
    /// `"<rname>:<pos>"`; for the others it is the single value stringified.
    pub oracle: String,
    /// Our candidate's rendering of the same field, formatted identically to `oracle` so the two
    /// are directly comparable by eye.
    pub ours: String,
}

/// Concordance summary between an oracle SAM and ours.
///
/// All counts are records. `compared` is the intersection of the two key sets, and every `*_match`
/// count is out of `compared`, NOT out of `oracle_records`: a read missing from one file cannot
/// mismatch on MAPQ, it is counted in `only_in_*` instead. So `oracle_records == our_records` and
/// `all_fields_match == compared` together are the "as good as this tool can see" condition.
#[derive(Debug, Default, Serialize)]
pub struct Report {
    /// Number of distinct keys parsed out of the oracle SAM, i.e. surviving primary records after
    /// headers, secondaries and supplementaries are dropped. Set once, up front, from the map's
    /// length. For a clean PE run this is 2 per pair.
    pub oracle_records: usize,
    /// The same count for our candidate SAM. `oracle_records != our_records` means one side emitted
    /// or dropped reads the other did not, and the difference shows up in `only_in_*`.
    pub our_records: usize,
    /// Keys present in BOTH files: the size of the intersection, and the denominator for every
    /// `*_match` field below. Accumulated one per matched key during the comparison loop.
    /// Identity: `compared + only_in_oracle == oracle_records`.
    pub compared: usize,
    /// Of `compared` records, how many agree on SAM column 2 (FLAG) exactly, as a whole bitfield.
    pub flag_match: usize,
    /// Of `compared` records, how many agree on RNAME and POS TOGETHER (SAM columns 3 and 4). A
    /// record that matches on only one of the two counts as a mismatch here.
    pub rname_pos_match: usize,
    /// Of `compared` records, how many agree on SAM column 5 (MAPQ). MAPQ is derived from the
    /// alignment scores of the whole candidate set, so it can differ even where RNAME/POS and
    /// CIGAR agree: this counter moving alone points at scoring or seed-set differences rather
    /// than at placement.
    pub mapq_match: usize,
    /// Of `compared` records, how many agree on SAM column 6 (CIGAR) as an exact string.
    pub cigar_match: usize,
    /// Of `compared` records, how many agree on all four verdicts at once. This is the headline
    /// number, and it is <= the minimum of the four `*_match` counts (a record must pass each).
    pub all_fields_match: usize,
    /// Keys in the oracle with no counterpart in ours: reads bwa-mem2 emitted a primary for and we
    /// did not. Counted in the main loop, where the lookup misses.
    pub only_in_oracle: usize,
    /// Keys in ours with no counterpart in the oracle: reads we emitted a primary for and bwa-mem2
    /// did not. Counted in a separate pass, since the main loop never visits these keys.
    pub only_in_ours: usize,
    /// Up to `max_examples` concrete divergences, in `HashMap` iteration order (so effectively
    /// arbitrary and not stable between runs). Empty when everything this tool inspects agrees.
    pub examples: Vec<Divergence>,
}

/// Parse primary alignments (skip `@` headers, secondary `0x100`, supplementary `0x800`), keyed
/// by QNAME plus the read-in-pair bit so PE mates do not collide.
///
/// The key is `"<qname>/<1|2>"`, derived from FLAG bit 0x80 ("last segment in template"), because
/// both mates of a pair share a QNAME by SAM's design. Single-end reads all key as `/1`.
///
/// Reads the whole file into memory (`read_to_string`), which is fine for the test-sized inputs
/// this is pointed at but not for a full 30x WGS SAM. Malformed fields parse as 0 or `*` rather
/// than erroring, on the reasoning that a diagnostic tool should still produce a report on damaged
/// input instead of refusing to run.
///
/// Invariant: at most one record per key survives. Duplicate primaries for one read would be a bug
/// in the producer, and the later one silently wins here.
///
/// # Parameters
///
/// * `path`: filesystem path to one uncompressed SAM text file (not BAM, not gzip: the bytes are
///   decoded as UTF-8). Supplied by the caller from argv. Must exist and be readable; its whole
///   size is allocated in memory, so keep it to test-scale inputs. Either file of the pair may be
///   passed, the function has no notion of oracle versus candidate.
///
/// # Returns
///
/// A map from `"<qname>/<1|2>"` to the [`Rec`] for that read, holding one entry per surviving
/// primary alignment. `Err` only for I/O and UTF-8 failures opening or reading the file; malformed
/// SAM content never errors.
pub fn parse_primary(path: &Path) -> std::io::Result<HashMap<String, Rec>> {
    // Entire SAM file as one UTF-8 string. This is the only fallible step.
    let text = std::fs::read_to_string(path)?;
    // Accumulator, and the return value. Invariant at the top of each iteration: it holds exactly
    // the primary records of the lines consumed so far, one per distinct read/mate key.
    let mut by_key = HashMap::new();
    // `line` is one SAM record (or header) with the newline stripped; iteration is file order,
    // which is discarded, the map is unordered.
    for line in text.lines() {
        if line.starts_with('@') || line.is_empty() {
            continue;
        }
        // SAM columns, in order: 1 QNAME, 2 FLAG, 3 RNAME, 4 POS, 5 MAPQ, 6 CIGAR. Anything past
        // column 6 (SEQ, QUAL, tags) is deliberately not read.
        // Tab-separated columns, consumed left to right; each `next()` advances one column, so the
        // order of these bindings IS the column order and must not be rearranged.
        let mut fields = line.split('\t');
        // Column 1 QNAME: the read name, shared by both mates of a pair.
        let qname = fields.next().unwrap_or("");
        // Column 2 FLAG: the bitfield, parsed as decimal. Tested immediately, so secondary and
        // supplementary lines cost only two columns of parsing.
        let flag: u32 = fields.next().unwrap_or("0").parse().unwrap_or(0);
        if flag & FLAG_SECONDARY != 0 || flag & FLAG_SUPPLEMENTARY != 0 {
            continue;
        }
        // Columns 3 to 6: RNAME (contig or `*`), POS (1-based, 0 if unplaced), MAPQ (Phred),
        // CIGAR (ops or `*`). The `unwrap_or` defaults are the SAM "absent" values, chosen so a
        // truncated line still compares sanely instead of claiming position 0 on a real contig.
        let rname = fields.next().unwrap_or("*").to_string();
        let pos: i64 = fields.next().unwrap_or("0").parse().unwrap_or(0);
        let mapq: u32 = fields.next().unwrap_or("0").parse().unwrap_or(0);
        let cigar = fields.next().unwrap_or("*").to_string();
        // Which mate of the pair this record is: 2 when FLAG bit 0x80 ("last segment") is set,
        // else 1. Single-end reads have the bit clear and so are all mate 1. This is the only
        // thing that keeps a pair's two records from overwriting each other under one QNAME.
        let mate = if flag & FLAG_LAST_SEGMENT != 0 { 2 } else { 1 };
        by_key.insert(
            format!("{qname}/{mate}"),
            Rec {
                flag,
                rname,
                pos,
                mapq,
                cigar,
            },
        );
    }
    Ok(by_key)
}

/// Compare two parsed SAMs, producing a concordance report (up to `max_examples` divergences).
///
/// `oracle` is the reference output (bwa-mem2) and `ours` the candidate; the roles are not
/// symmetric, since iteration is over `oracle` and the `only_in_*` counts are named accordingly.
/// `max_examples` caps the `examples` vector so a wholesale divergence does not produce a report
/// the size of the input.
///
/// Each divergent record contributes exactly ONE example, attributed to the first differing field
/// in the priority order RNAME/POS, CIGAR, MAPQ, FLAG. That order is deliberate: a wrong position
/// explains a wrong CIGAR, which explains a wrong MAPQ, so reporting the root-most field keeps the
/// example list pointed at causes rather than consequences. The per-field `*_match` counters are
/// unaffected and still count every field independently.
///
/// # Parameters
///
/// * `oracle`: the trusted side, bwa-mem2's primaries as returned by [`parse_primary`]. Iterated,
///   so it drives which keys are examined and it alone determines `only_in_oracle`.
/// * `ours`: the candidate side, same shape, probed by key only. Swapping the two arguments does
///   not produce a mirrored report, it produces a wrong one (the `only_in_*` labels would lie).
/// * `max_examples`: hard cap on `Report::examples`, in records. 0 disables examples entirely and
///   leaves every count intact. Supplied by the caller (`MAX_EXAMPLES` = 20 in the `sam_diff`
///   binary) purely to bound output size; it never affects any counter.
///
/// # Returns
///
/// A fully populated [`Report`]. Never fails, and never panics on mismatched or empty inputs.
pub fn compare(
    oracle: &HashMap<String, Rec>,
    ours: &HashMap<String, Rec>,
    max_examples: usize,
) -> Report {
    // The accumulator built across the whole function. The two record totals are final from here;
    // every other counter starts at 0 and only grows. Invariant at the top of each loop iteration:
    // all counters reflect exactly the oracle keys visited so far, and
    // `compared + only_in_oracle` equals the number of keys visited.
    let mut report = Report {
        oracle_records: oracle.len(),
        our_records: ours.len(),
        ..Default::default()
    };
    // `key` is `"<qname>/<mate>"` and `oracle_rec` the trusted record for it. Iteration order is
    // `HashMap` order, hence arbitrary: nothing below may depend on it except which divergences
    // happen to land in the capped `examples` list.
    for (key, oracle_rec) in oracle {
        // Our record for the same read and mate, or nothing: a read the oracle placed and we did
        // not. There is no field to compare in that case, so it is counted and skipped.
        let Some(our_rec) = ours.get(key) else {
            report.only_in_oracle += 1;
            continue;
        };
        report.compared += 1;

        // ---- Per-field verdicts. RNAME and POS are one verdict: a position is the pair ----
        // The four verdicts for THIS record, all independent of each other. Each is true when the
        // two sides agree; `pos_ok` is the conjunction of RNAME and POS because a location is the
        // pair, not either half. All four are reused twice below: once for the counters, once to
        // decide whether an example is owed.
        let flag_ok = oracle_rec.flag == our_rec.flag;
        let pos_ok = oracle_rec.rname == our_rec.rname && oracle_rec.pos == our_rec.pos;
        let mapq_ok = oracle_rec.mapq == our_rec.mapq;
        let cigar_ok = oracle_rec.cigar == our_rec.cigar;
        report.flag_match += usize::from(flag_ok);
        report.rname_pos_match += usize::from(pos_ok);
        report.mapq_match += usize::from(mapq_ok);
        report.cigar_match += usize::from(cigar_ok);

        // ---- At most one example per divergent record, attributed to the root-most field ----
        if flag_ok && pos_ok && mapq_ok && cigar_ok {
            report.all_fields_match += 1;
        } else if report.examples.len() < max_examples {
            // The one field this record's example is attributed to, plus both sides rendered as
            // strings. The if-chain order RNAME/POS, CIGAR, MAPQ, FLAG is the causal order: the
            // first failing test wins and the rest are not reported for this record.
            let (field, oracle_value, our_value) = if !pos_ok {
                (
                    "RNAME/POS",
                    format!("{}:{}", oracle_rec.rname, oracle_rec.pos),
                    format!("{}:{}", our_rec.rname, our_rec.pos),
                )
            } else if !cigar_ok {
                ("CIGAR", oracle_rec.cigar.clone(), our_rec.cigar.clone())
            } else if !mapq_ok {
                (
                    "MAPQ",
                    oracle_rec.mapq.to_string(),
                    our_rec.mapq.to_string(),
                )
            } else {
                (
                    "FLAG",
                    oracle_rec.flag.to_string(),
                    our_rec.flag.to_string(),
                )
            };
            report.examples.push(Divergence {
                key: key.clone(),
                field: field.to_string(),
                oracle: oracle_value,
                ours: our_value,
            });
        }
    }
    // ---- Reads present only in our output: the loop above could not have seen them ----
    report.only_in_ours = ours.keys().filter(|k| !oracle.contains_key(*k)).count();
    report
}
