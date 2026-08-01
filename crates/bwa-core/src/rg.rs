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
//!
//! # Rust mechanics used in this file
//!
//! The interesting part of this file is that it holds state shared by every worker thread at once.
//! Rust does not allow an ordinary mutable global, so the two `static`s below use types that make
//! concurrent access safe by construction. That is what most of this table is about.
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `static` | one value existing for the whole run of the program, at a fixed address, shared by every thread. Unlike a C global it may not be plainly mutable, because two threads writing it at once would be a data race. |
//! | `OnceLock<T>` | a slot that starts empty and can be filled exactly once, safely, from any thread. After it is filled, reading costs no lock at all, which is why the read group can be looked up on the per-record hot path. A second write is refused rather than overwriting. |
//! | `AtomicBool` | a boolean that can be read and written from several threads without tearing or locking. |
//! | `Ordering::Relaxed` | the weakest guarantee an atomic can be given: this value's own reads and writes are coherent, but nothing else is ordered around it. Correct here because the flag is written once before any worker starts and only read afterwards. |
//! | `&mut Vec<u8>` | a BORROWED, exclusive handle on somebody else's growable byte buffer. Exclusive means the compiler guarantees nobody else can read or write it while this borrow lives, so appending in place is safe. Nothing is copied and nothing is returned. |
//! | `Option<&str>` | either borrowed text or nothing. `&str` is a window onto text somebody else owns, as opposed to `String`, which owns its bytes. |
//! | `&'static str` | borrowed text guaranteed to live as long as the program. Returning one is only possible because the `OnceLock` never moves or frees its contents once set. |
//! | `Result<String, String>` | succeeds with a `String` or fails with a `String`. Both halves are text here because the error is bwa's own wording, reproduced verbatim for parity. |
//! | `let Some(x) = ... else { ... }` | the "let-else" form: bind `x` and continue, or run the `else` block, which must leave the function. It is how a missing value becomes an early return without nesting the rest of the function inside an `if`. |
//! | `.into()` | converts a value into whatever type the context requires, here a borrowed `&str` literal into an owned `String`. The target type is inferred from the signature, never written. |
//! | `.chars()` | walks text one CHARACTER at a time, as opposed to `.bytes()`, which walks raw bytes. The two differ for non-ASCII text. |
//! | `.take_while(...)` | keeps yielding elements while a test holds, then stops for good at the first failure. It is the scan-until-separator idiom. |
//! | `\|&c\| c != '\t'` | an anonymous inline function used as that test. The `&` unwraps the reference the walk yields, so `c` is the character itself. |
//! | `let _ = expr;` | deliberately discard a returned value. Written out because Rust warns about ignoring a `Result` by accident; the `_` says the omission is intentional. |
//! | `&line[a..b]` | a slice: a window onto part of an existing string, with no copy. `a..b` is a half-open range, including `a` and excluding `b`. |

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
    // Two independent reasons to write nothing, checked separately: the flag is off, or this
    // particular read had no comment.
    //
    // Rust: `out` is borrowed exclusively (`&mut`), so this function appends to the caller's buffer
    // in place and returns nothing. The caller sees the change without any value being handed back.
    if !copy_comment() {
        return;
    }
    // Rust: unwraps the `Option`, running the body only when there is a comment, with `c` bound to
    // it. `.as_bytes()` reinterprets the text as raw bytes without copying, because the SAM record
    // is being assembled as bytes rather than as text.
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
    // Rust: `.get()` peeks into the slot, giving `Some` once it has been filled and `None` before.
    // `.map(String::as_str)` converts what is inside, from a reference to the owned `String` into a
    // plain borrowed `&str`, without copying the text. Passing `String::as_str` by name instead of
    // writing `|s| s.as_str()` is the same thing spelled shorter.
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
    //
    // Rust: `with_capacity` reserves the buffer up front instead of growing it repeatedly. `mut` is
    // required on both bindings; without it the compiler refuses every push and every increment,
    // because a binding is read-only unless declared otherwise.
    let mut out = String::with_capacity(s.len());
    let mut pos = 0;
    // A hand-written index loop rather than an iterator, because an escape consumes TWO bytes and
    // an iterator walking one at a time cannot express that without extra state.
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
                //
                // Rust: `_` is the catch-all pattern, "any other value", and `{}` is an empty body.
                // The arm exists because a `match` must cover every possibility to compile; writing
                // it out is what documents that dropping unknown escapes is deliberate rather than
                // an oversight.
                _ => {}
            }
            pos += 1;
        } else {
            // An ordinary byte, copied through. `as char` widens the byte to a character; this is
            // only correct because a `-R` line is ASCII, and it is what keeps the C's byte-at-a-time
            // scan reproducible here.
            out.push(bytes[pos] as char);
            pos += 1;
        }
    }
    // No semicolon: the accumulated string is the return value.
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
    //
    // Rust: the "let-else" form. On success `id_marker_pos` is bound and the function continues at
    // the same indentation; on failure the `else` block runs and must leave the function. This is
    // what keeps the rest of the body from being nested one level deeper inside an `if`.
    let Some(id_marker_pos) = line.find("\tID:") else {
        return Err("no ID at the read group line".into());
    };
    // Everything from the first character of the ID to the end of the line; the ID is its prefix up
    // to the next field separator.
    let after_marker = &line[id_marker_pos + 4..];
    // The read-group ID itself, i.e. the `foo` in `ID:foo`. A newline terminates it as well as a
    // tab, matching the C's scan, so a stray embedded newline truncates rather than corrupting the
    // header.
    //
    // Rust: read as "walk the characters, keep them while neither separator is seen, gather what
    // survives into a `String`". `.take_while` stops permanently at the first character that fails
    // the test, so this yields the prefix, not every non-separator character in the rest of the
    // line. The `: String` annotation is what tells `.collect()` which container to build.
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
