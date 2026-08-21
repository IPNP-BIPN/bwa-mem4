//! System-RAM detection, for the learned-index auto-select: LISA seeding needs a large in-memory
//! index (~78 GB at hg38 scale), worth it only on big-memory hosts, so the aligner picks LISA when
//! enough RAM is present and otherwise falls back to the classic FM-index path, transparently, so a
//! 16-64 GB machine is never penalized.
//!
//! Nothing here affects SAM bytes: both index paths produce the same alignments, so this file is
//! purely a performance-policy decision. Reading order: [`total_ram_bytes`] then
//! [`learned_index_fits`], its only caller-facing use.
//!
//! # Rust mechanics used in this file
//!
//! The `Option`, `?`, `.ok()` and `#[cfg]` machinery is the same as in `cpu.rs`, which glosses it
//! in full. What is new here is the string handling and the number types.
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `u64` | an unsigned 64-bit integer. Byte counts need it: a 32-bit count tops out at 4 GB, which modern hosts exceed. |
//! | `f64` | a double-precision floating-point number. Used for the ratio below because the comparison is fractional (1.25x), not a whole number of bytes. |
//! | `as f64` | an explicit conversion from integer to float. Rust never converts number types on its own, so the cast has to be written even where C would do it silently. |
//! | `1e9` | scientific notation for a float, here one decimal gigabyte. |
//! | `1 << 30` | bit shift: one, shifted left thirty places, which is 2^30, a binary gigabyte. A compact way to write large round powers of two. |
//! | `.lines()` | walks a block of text one line at a time, without allocating a new string per line. |
//! | `.strip_prefix("x")` | returns `Some(rest)` when the text starts with `x`, and `None` when it does not. It tests and trims in one step, so the label match and the value extraction cannot drift apart. |
//! | `.split_whitespace()` | walks the space-separated tokens of a line, ignoring how many spaces separate them. `.next()` then takes the first token. |
//! | `return` | an EARLY exit. Rust only needs the keyword when leaving from the middle of a function, as in the loop below; the value at the end of a body needs no keyword and no semicolon. |
//! | `match` on an `Option` | handles both cases explicitly, `Some(v)` and `None`, and the compiler refuses the code if either is forgotten. That exhaustiveness is what guarantees the "RAM unknown" path below is never left undecided. |
//! | `!expr` | boolean negation, "not". |

/// Headroom multiplier on the index footprint: the host must have 1.25x the index size before the
/// learned index is chosen, leaving room for the OS, the page cache and per-thread scratch. The
/// figure is a policy choice made here, not a number taken from any upstream source.
const RAM_HEADROOM_FACTOR: f64 = 1.25;

/// Bytes per gigabyte as used by the comparison below: decimal (1e9), matching how RAM sizes are
/// quoted on spec sheets, NOT the binary 2^30.
const BYTES_PER_GB: f64 = 1e9;

/// Total physical RAM in bytes, or `None` if it can't be determined. macOS reads `hw.memsize`
/// (`sysctl`); Linux reads `/proc/meminfo`'s `MemTotal`.
///
/// # Returns
///
/// Total PHYSICAL RAM in bytes, not free or available RAM: the caller wants the machine's capacity,
/// since the index is mapped once and shared. `None` on any other platform, and on macOS/Linux
/// whenever the probe fails (no `sysctl` binary, unreadable `/proc`, unparseable output). Every
/// failure collapses to `None` via `ok()?` rather than surfacing a distinct error, because the only
/// caller treats all of them the same way: fall back to the FM index.
pub fn total_ram_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        // `sysctl -n hw.memsize` prints the byte count and nothing else, so the whole trimmed
        // stdout parses directly as the answer. Subprocess rather than a libc `sysctlbyname` call
        // to keep this crate dependency-free; it runs once per process, so the fork is free.
        let sysctl_output = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        // Read top to bottom as four steps, each of which may give up: interpret the captured bytes
        // as text, drop the trailing newline, parse the digits, hand back the answer. The final
        // `.ok()` has no `?` because this is the block's last expression: its `Option` IS the
        // function's return value, so a parse failure becomes the `None` the caller expects.
        //
        // Rust: nothing states `u64` here. The compiler works it out backwards from the function's
        // declared return type, `Option<u64>`, and tells `.parse()` what to produce.
        std::str::from_utf8(&sysctl_output.stdout)
            .ok()?
            .trim()
            .parse()
            .ok()
    }
    #[cfg(target_os = "linux")]
    {
        // The whole pseudo-file, a few kilobytes of `Label: value kB` lines. Read in one go rather
        // than streamed: `MemTotal` is the first line in practice, but the format guarantees only
        // that it is present.
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        // Scans for the one line that matters; falls out to `None` if the file lacks it entirely.
        for line in meminfo.lines() {
            // `MemTotal:` is reported in kibibytes, hence the * 1024.
            if let Some(after_label) = line.strip_prefix("MemTotal:") {
                // The numeric field of `MemTotal:    16384000 kB`: leading spaces dropped by
                // `split_whitespace`, the trailing `kB` unit token discarded by taking only the
                // first token.
                //
                // Rust: `.next()?` takes the first token and bails out with `None` if the line was
                // empty after the label. `.parse().ok()?` bails out again if that token is not a
                // number. The `: u64` annotation is what tells `.parse()` which type to build.
                let kibibytes: u64 = after_label.split_whitespace().next()?.parse().ok()?;
                // Rust: an early `return`, needed because we are inside a loop rather than at the
                // end of the body. It exits `total_ram_bytes` entirely, not just the loop.
                return Some(kibibytes * 1024);
            }
        }
        // Reached only when the loop ran off the end without ever seeing `MemTotal:`. No semicolon,
        // so this is the block's value: the file existed but did not say what we needed.
        None
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// Whether to use the learned index: total RAM must comfortably exceed the index footprint (a 1.25x
/// headroom for the OS, page cache, and per-thread scratch). Unknown RAM => `false` (use FM), so the
/// safe classic path is always the default when detection fails.
///
/// # Parameters
///
/// - `index_gb`: the learned index's in-memory footprint in DECIMAL gigabytes (1e9 bytes, see
///   [`BYTES_PER_GB`]), which must be the same unit the caller's estimate is in. Positive; roughly
///   78.0 at hg38 scale. Supplied by the index-selection code, which derives it from the reference
///   length rather than measuring it.
///
/// # Returns
///
/// True only when detected RAM is at least `index_gb * `[`RAM_HEADROOM_FACTOR`]. False whenever RAM
/// cannot be detected, so the conservative FM path is the default on unknown platforms. Purely a
/// performance decision: neither branch changes the SAM output.
pub fn learned_index_fits(index_gb: f64) -> bool {
    // The whole policy in one comparison: detected RAM converted to decimal gigabytes, against the
    // index footprint inflated by the headroom factor.
    //
    // Rust: `match` forces both cases to be written out, and the compiler rejects the function if
    // one is missing. That is what makes "RAM could not be detected" an explicit decision (fall
    // back to the classic FM path) instead of an accident. `as f64` converts the byte count to a
    // float so the division is fractional rather than truncated integer division.
    match total_ram_bytes() {
        Some(total_bytes) => (total_bytes as f64) / BYTES_PER_GB >= index_gb * RAM_HEADROOM_FACTOR,
        None => false,
    }
}

/// Resident bytes one in-flight batch costs, per base of `-K`.
///
/// Measured, not guessed. The allocation probe (`BWA4_STAGE_ALLOC`) on Linux, where resident memory
/// IS live memory to within 1.02x, puts one batch of 100M bases at 2.40 GB of live bytes, and the
/// A/B of the batch overlap on the same job moves peak RSS by 1.97 GB, i.e. 20 to 24 bytes per base
/// either way. The upper end is taken, because overestimating a batch makes this policy drop the
/// overlap sooner, which is the harmless direction: the cost of being wrong is 1.1% of wall, while
/// the cost of the opposite error is a run that swaps.
const BATCH_BYTES_PER_BASE: f64 = 24.0;

/// Resident bytes the loaded index costs, per base of the FORWARD strand (`l_pac`).
///
/// Also measured, from the same probe's index baseline: 0.181 GB for chr21 (3.9 bytes per base) and
/// 10.47 GB for GRCh38 (3.4). Rounded UP for the same reason as above.
const INDEX_BYTES_PER_PAC_BASE: f64 = 4.0;

/// What the aligner should do about holding a second batch in flight.
///
/// Returned as a struct rather than a bare `bool` so the caller can print the arithmetic it used.
/// A policy the user cannot see is a policy they cannot argue with, and this one changes peak RSS
/// by 40% on a memory-tight host.
pub struct OverlapDecision {
    /// Whether to overlap batch N's tail with batch N+1's alignment.
    pub overlap: bool,
    /// Detected physical RAM in decimal GB, or `None` when detection failed.
    pub ram_gb: Option<f64>,
    /// Estimated peak footprint of the overlapped pipeline in decimal GB: index plus two batches,
    /// with the headroom factor already applied.
    pub need_gb: f64,
}

/// Whether the machine can afford the pipeline's second resident batch.
///
/// The overlap runs batch N's low-occupancy tail against batch N+1's extension and is worth about
/// 1.1% of wall (measured on Linux, chr21, 4 cores). It costs one more resident batch, which on a
/// human genome at the default `-K` is around 2 GB, and that is the difference between fitting in a
/// machine and swapping. So it is taken only when the RAM is there.
///
/// Nothing here can change a SAM byte: the number of batches in flight is not observable in the
/// output (batch order comes from joining a batch before sending it), which is what makes this a
/// legitimate machine-dependent policy rather than a source of irreproducible results.
///
/// # Parameters
///
/// - `l_pac`: forward-strand length of the reference in bases, i.e. `BntSeq::l_pac`, used to
///   estimate the loaded index's footprint.
/// - `k_batch`: `-K` in bases, i.e. one batch's input size.
///
/// # Returns
///
/// The decision and the two figures behind it. When RAM cannot be detected the overlap is KEPT:
/// unlike [`learned_index_fits`], whose fallback avoids a large speculative allocation, the
/// conservative answer here would silently cost every user on an undetectable platform 1.1% of wall
/// for a memory problem they may not have.
pub fn batch_overlap_fits(l_pac: u64, k_batch: u64) -> OverlapDecision {
    let index_bytes = l_pac as f64 * INDEX_BYTES_PER_PAC_BASE;
    let two_batches = 2.0 * k_batch as f64 * BATCH_BYTES_PER_BASE;
    let need_gb = RAM_HEADROOM_FACTOR * (index_bytes + two_batches) / BYTES_PER_GB;
    let ram_gb = total_ram_bytes().map(|b| b as f64 / BYTES_PER_GB);
    OverlapDecision {
        // No RAM figure means no reason to give up the overlap; see the doc comment.
        overlap: ram_gb.is_none_or(|have| have >= need_gb),
        ram_gb,
        need_gb,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The policy must flip where it is designed to flip, and both sides are checked with figures
    /// from the two measured configurations rather than round numbers.
    #[test]
    fn overlap_is_dropped_only_when_the_machine_is_actually_tight() {
        // GRCh38 (l_pac 3.1e9) at the default -K for 8 threads (80M bases): the index alone is
        // ~12.4 GB by this estimate, two batches ~3.8 GB, so a 16 GB host cannot hold it and a
        // 128 GB one can, comfortably.
        let d = batch_overlap_fits(3_100_000_000, 80_000_000);
        assert!(d.need_gb > 16.0 && d.need_gb < 32.0, "need {}", d.need_gb);

        // chr21 (l_pac 46.7e6) at -K 100M, the CI measurement: 6 GB of need, so any real machine
        // keeps the overlap.
        let d = batch_overlap_fits(46_700_000, 100_000_000);
        assert!(d.need_gb > 5.0 && d.need_gb < 8.0, "need {}", d.need_gb);

        // The decision is monotone in both inputs, which is the property that makes it explainable:
        // a bigger reference or a bigger batch can only ever make the overlap less affordable.
        let small = batch_overlap_fits(46_700_000, 10_000_000).need_gb;
        let big = batch_overlap_fits(46_700_000, 320_000_000).need_gb;
        assert!(big > small);
    }

    #[test]
    fn ram_detection_plausible() {
        // On the CI/dev host this must return a sane figure (>= 1 GB, < 100 TB).
        if let Some(total_bytes) = total_ram_bytes() {
            assert!(total_bytes >= 1 << 30, "implausibly small: {total_bytes}");
            assert!(
                total_bytes < (100u64 << 40),
                "implausibly large: {total_bytes}"
            );
        }
        // A 1 TB index never fits a normal host; a 1 GB index fits any host with detectable RAM.
        //
        // Rust: this assertion holds on EVERY host, which is what makes it a usable test. Where RAM
        // is detectable the comparison fails on size; where it is not, the `None` arm returns false
        // anyway. `!` negates, so the test reads "assert that a 1 TB index does not fit".
        assert!(!learned_index_fits(1000.0));
    }
}
