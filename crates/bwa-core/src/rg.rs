//! Read-group state for `-R`, mirroring bwa's global `bwa_rg_id`.
//!
//! bwa keeps the parsed `@RG` ID in a process-wide global (`bwa_rg_id` in `bwa.cpp`) and every
//! record emitter appends `\tRG:Z:<id>` when it is non-empty. We reproduce that shape rather than
//! threading the id through a dozen signatures, so the three emission sites (SE tags, PE tags and
//! the unmapped-record writer) stay in step exactly as the C does.
//!
//! # Glossary
//!
//! | Term | Plain language |
//! |------|----------------|
//! | `@RG` | SAM header line describing one "read group": which library/sample/lane a read came from |
//! | `RG:Z:<id>` | the per-record tag pointing back at that header line |
//! | comment | whatever followed the first whitespace on the FASTQ header line |
//!
//! Reading order: [`set_rg`] parses and installs the id, [`rg_id`]/[`append_rg_tag`] read it back
//! at emission time, [`escape`] is the shared backslash expander both paths use.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// bwa's `bwa_set_rg` rejects an ID that does not fit its `char[256]` buffer, i.e. more than 255
/// characters plus the NUL terminator (`bwa.cpp`). The comparison below keeps the C's `+ 1` for the
/// terminator so the boundary case matches exactly.
const MAX_RG_ID_BUFFER: usize = 256;

/// The `@RG` ID parsed out of the `-R` line, e.g. `foo` from `@RG\tID:foo\tSM:bar`.
///
/// Write-once at CLI-parse time, then read by every record emitter on every worker thread, which is
/// exactly the `OnceLock` access pattern: no lock on the read path. Unset (`None`) means `-R` was
/// not given and no `RG:Z:` tag is emitted. A second `set_rg` call is silently ignored rather than
/// panicking (see the discarded result in [`set_rg`]).
static RG_ID: OnceLock<String> = OnceLock::new();

/// Whether `-C` was given, i.e. whether to copy each read's FASTQ comment onto its SAM record.
///
/// Set once before alignment starts and only read afterwards, so `Relaxed` ordering suffices: there
/// is no other state whose visibility has to be ordered against it.
static COPY_COMMENT: AtomicBool = AtomicBool::new(false);

/// `-C`: append the FASTA/FASTQ comment at the very end of every SAM record. bwa keeps this in
/// `aux.copy_comment` (and frees each read's comment when it is off), so it is process-wide state
/// like the read group above.
///
/// # Parameters
///
/// - `on`: true if `-C` appeared on the command line. Supplied once by `bwa-cli::cmd_mem` before
///   any worker thread starts; calling it mid-run would race the emitters and is not done.
pub fn set_copy_comment(on: bool) {
    COPY_COMMENT.store(on, Ordering::Relaxed);
}

/// Whether `-C` was given.
///
/// # Returns
///
/// The current [`COPY_COMMENT`] flag; false until [`set_copy_comment`] says otherwise.
pub fn copy_comment() -> bool {
    COPY_COMMENT.load(Ordering::Relaxed)
}

/// Append `\t<comment>` when `-C` is on and the read carried one. bwa emits it last, after every
/// tag including SA:Z/XA:Z (`mem_aln2sam`: `if (s->comment) { kputc('\t'); kputs(s->comment); }`).
///
/// # Parameters
///
/// - `out`: the SAM record being built, appended to in place. Must already hold every mandatory
///   field and every tag, since the comment goes last; the trailing newline is added by the caller
///   afterwards.
/// - `comment`: whatever followed the first whitespace on this read's FASTQ header line, or `None`
///   when the header had no comment. Supplied per read by the SAM writer in `bwa-io`.
///
/// Writes nothing at all unless `-C` is on AND the read carried a comment, so both a `None` and a
/// disabled flag leave `out` byte-for-byte untouched.
pub fn append_comment(out: &mut Vec<u8>, comment: Option<&str>) {
    if !copy_comment() {
        return;
    }
    if let Some(c) = comment {
        out.push(b'\t');
        out.extend_from_slice(c.as_bytes());
    }
}

/// The read-group ID to stamp on every record, or `None` when `-R` was not given.
///
/// # Returns
///
/// A `'static` borrow of [`RG_ID`], valid for the process lifetime because a `OnceLock`'s contents
/// are never moved or dropped once set. Read on the hot emission path, hence no allocation.
pub fn rg_id() -> Option<&'static str> {
    RG_ID.get().map(String::as_str)
}

/// Append `\tRG:Z:<id>` if a read group is set. bwa emits it right after `AS`/`XS` and before
/// `SA:Z` (`bwamem.cpp`: `if (bwa_rg_id[0]) { kputsn("\tRG:Z:", 6, str); ... }`).
///
/// # Parameters
///
/// - `out`: the SAM record under construction, appended to in place. Must be positioned just after
///   the `AS`/`XS` tags: the tag order is part of byte-parity with bwa-mem2, so calling this
///   earlier or later changes the output even though the tag set is identical.
///
/// Leaves `out` untouched when no read group is set.
pub fn append_rg_tag(out: &mut Vec<u8>) {
    if let Some(id) = rg_id() {
        out.extend_from_slice(b"\tRG:Z:");
        out.extend_from_slice(id.as_bytes());
    }
}

/// Expand bwa's backslash escapes in place-equivalent fashion (`bwa_escape`): `\t`, `\n`, `\r` and
/// `\\`. Any other escaped character is dropped, exactly as the C does.
///
/// # Parameters
///
/// - `s`: the raw `-R` argument as the shell handed it over, e.g. `@RG\tID:foo\tSM:bar` with a
///   literal backslash-t rather than a tab (the shell does not expand these, which is the whole
///   reason bwa does it itself). Any string is accepted; a trailing lone backslash is copied
///   through verbatim because the `pos + 1 < len` guard fails on it.
///
/// # Returns
///
/// A new `String` with the four recognised escapes expanded. Length is at most `s.len()`, which is
/// what the `with_capacity` reserves, so the buffer never reallocates.
pub fn escape(s: &str) -> String {
    let bytes = s.as_bytes();
    // Accumulator: the expansion of everything before `pos`. Invariant at the top of each
    // iteration: `out` holds the fully expanded translation of `bytes[..pos]`, and `pos` sits on
    // the first byte not yet consumed. Each iteration consumes either 1 byte (ordinary) or 2 (a
    // backslash and the letter it escapes), so `pos` strictly increases and the loop terminates.
    let mut out = String::with_capacity(s.len());
    let mut pos = 0;
    while pos < bytes.len() {
        // A backslash with at least one byte after it: `pos` advances onto the escaped letter, so
        // from here `bytes[pos]` is the letter, not the backslash. A backslash in final position
        // fails this test and falls through to be copied literally.
        if bytes[pos] == b'\\' && pos + 1 < bytes.len() {
            pos += 1;
            match bytes[pos] {
                b't' => out.push('\t'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b'\\' => out.push('\\'),
                // bwa writes nothing for an unknown escape.
                _ => {}
            }
            pos += 1;
        } else {
            out.push(bytes[pos] as char);
            pos += 1;
        }
    }
    out
}

/// Parse and install a `-R` read-group line, returning the escaped line to put in the header.
/// Port of `bwa_set_rg`: the line must start with `@RG` and carry a `\tID:` field (<= 255 chars).
///
/// # Parameters
///
/// - `s`: the unescaped `-R` argument, e.g. `@RG\tID:foo\tSM:bar`. Preconditions, each of which is
///   checked and reported rather than assumed: it must begin with the literal `@RG`, and after
///   escape expansion it must contain a `\tID:` field whose value is at most 255 characters.
///   Supplied once by `bwa-cli::cmd_mem` at startup.
///
/// # Returns
///
/// `Ok(line)` with the escape-expanded line, which the caller writes verbatim into the SAM header;
/// the extracted ID itself is not returned but installed in [`RG_ID`] as a side effect, and is read
/// back through [`rg_id`]. `Err(msg)` carries bwa's own wording for the three failure modes, so the
/// CLI's stderr matches the C's byte for byte.
pub fn set_rg(s: &str) -> Result<String, String> {
    if !s.starts_with("@RG") {
        return Err("the read group line is not started with @RG".into());
    }
    // The header line as it will actually be printed: real tabs, not backslash-t. Everything below
    // searches THIS string, not `s`, so the `\tID:` marker means a genuine tab byte.
    let line = escape(s);
    // Offset of the literal "\tID:" marker; +4 skips those four bytes to the id itself.
    let Some(id_marker_pos) = line.find("\tID:") else {
        return Err("no ID at the read group line".into());
    };
    // Everything from the first character of the ID to the end of the line; the ID is its prefix up
    // to the next field separator.
    let after_marker = &line[id_marker_pos + 4..];
    // The read-group ID itself, i.e. the `foo` in `ID:foo`. A newline terminates it as well as a
    // tab, matching the C's scan, so a stray embedded newline truncates rather than corrupting the
    // header.
    let id: String = after_marker
        .chars()
        .take_while(|&c| c != '\t' && c != '\n')
        .collect();
    if id.len() + 1 > MAX_RG_ID_BUFFER {
        return Err("@RG:ID is longer than 255 characters".into());
    }
    // Result deliberately discarded: `set` fails only if the id was already installed, and bwa's
    // single `-R` means that cannot legitimately happen. First writer wins if it ever does.
    let _ = RG_ID.set(id);
    Ok(line)
}
